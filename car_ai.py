"""Small online policy-gradient driver for the procedural car game."""

from pathlib import Path
from array import array
from dataclasses import dataclass, field
import tempfile
from typing import List, Sequence, Tuple
import uuid
import warnings

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from car_simulation import Observation


@dataclass
class EpisodeRollout:
	"""Compact CPU storage for one completed episode's training experience."""

	observation_size: int
	observations: array = field(default_factory=lambda: array("f"))
	actions: array = field(default_factory=lambda: array("b"))
	rewards: array = field(default_factory=lambda: array("f"))
	log_probabilities: array = field(default_factory=lambda: array("f"))

	def append(
		self,
		observation: Observation,
		action: int,
		reward: float,
		log_probability: float,
	) -> None:
		if len(observation) != self.observation_size:
			raise ValueError("observation size does not match the rollout")
		if not 0 <= action <= 8:
			raise ValueError("action must be between 0 and 8")
		if not torch.isfinite(torch.tensor((reward, log_probability))).all().item():
			raise ValueError("reward and log_probability must be finite")
		self.observations.extend(observation)
		self.actions.append(action)
		self.rewards.append(reward)
		self.log_probabilities.append(log_probability)

	def __len__(self) -> int:
		return len(self.actions)


class DrivingPolicy(nn.Module):
	"""Map normalized track/traffic observations to nine driving actions."""

	def __init__(self, observation_size: int, action_count: int = 9) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(observation_size, 256),
			nn.Tanh(),
			nn.Linear(256, 256),
			nn.Tanh(),
			nn.Linear(256, action_count),
		)

	def forward(self, observation: Tensor) -> Tensor:
		return self.layers(observation)


class OnlineDriver:
	"""Collect complete runs and update a stochastic policy using REINFORCE."""

	def __init__(
		self,
		observation_size: int,
		checkpoint_path: Path | str = Path(__file__).with_name("car_ai_checkpoint_v4.pt"),
		device: torch.device | str | None = None,
		learning_rate: float = 3e-4,
		gamma: float = 0.99,
	) -> None:
		self.device = torch.device(
			device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
		)
		self.policy = DrivingPolicy(observation_size).to(self.device)
		self.optimizer = torch.optim.AdamW(
			self.policy.parameters(),
			lr=learning_rate,
			weight_decay=1e-4,
		)
		self.checkpoint_path = Path(checkpoint_path)
		self.gamma = gamma
		self._log_probabilities: List[Tensor] = []
		self._rewards: List[float] = []
		self.episodes_trained = 0
		self.load_checkpoint()

	def choose_action(self, observation: Observation) -> Tuple[int, Tensor]:
		"""Sample a driving action and return its differentiable log probability."""
		state = torch.tensor(observation, dtype=torch.float32, device=self.device)
		distribution = Categorical(logits=self.policy(state))
		action = distribution.sample()
		return int(action.item()), distribution.log_prob(action)

	def record(self, log_probability: Tensor, reward: float) -> None:
		"""Retain the policy graph and reward for one frame of the current run."""
		if not torch.isfinite(log_probability).all().item() or not torch.isfinite(
			torch.tensor(reward)
		).item():
			raise ValueError("log_probability and reward must be finite")
		self._log_probabilities.append(log_probability)
		self._rewards.append(float(reward))

	def finish_episode(self) -> float | None:
		"""Apply one discounted REINFORCE update and persist the updated policy."""
		if not self._rewards:
			return None
		if len(self._rewards) != len(self._log_probabilities):
			raise RuntimeError("episode rewards and action probabilities are misaligned")

		returns: List[float] = []
		discounted_return = 0.0
		for reward in reversed(self._rewards):
			discounted_return = reward + self.gamma * discounted_return
			returns.append(discounted_return)
		returns.reverse()

		return_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
		return_tensor = (return_tensor - return_tensor.mean()) / (
			return_tensor.std(unbiased=False) + 1e-6
		)
		log_prob_tensor = torch.stack(self._log_probabilities)
		loss = -(log_prob_tensor * return_tensor).mean()

		self.optimizer.zero_grad(set_to_none=True)
		loss.backward()
		nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
		self.optimizer.step()
		self.episodes_trained += 1
		self._log_probabilities.clear()
		self._rewards.clear()
		self.save_checkpoint()
		return float(loss.detach().cpu().item())

	def save_checkpoint(self) -> None:
		"""Save a rotating generation to avoid replacing OneDrive-locked files."""
		self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"policy": self.policy.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"episodes_trained": self.episodes_trained,
		}
		generation_path = self.checkpoint_path.with_name(
			f"{self.checkpoint_path.stem}_run_{self.episodes_trained:06d}_"
			f"{uuid.uuid4().hex[:8]}{self.checkpoint_path.suffix}"
		)
		temporary_path: Path | None = None
		try:
			with tempfile.NamedTemporaryFile(
				prefix=f"{self.checkpoint_path.name}.",
				suffix=".tmp",
				dir=self.checkpoint_path.parent,
				delete=False,
			) as temporary_file:
				temporary_path = Path(temporary_file.name)
			torch.save(payload, temporary_path)
			temporary_path.replace(generation_path)
		finally:
			if temporary_path is not None:
				try:
					temporary_path.unlink(missing_ok=True)
				except OSError as exc:
					warnings.warn(
						f"Could not remove temporary checkpoint {temporary_path}: {exc}",
						RuntimeWarning,
						stacklevel=2,
					)

		generations = self._checkpoint_generations()
		for old_checkpoint in generations[5:]:
			try:
				old_checkpoint.unlink()
			except OSError as exc:
				warnings.warn(
					f"Could not remove old checkpoint {old_checkpoint}: {exc}",
					RuntimeWarning,
					stacklevel=2,
				)

	def _checkpoint_generations(self) -> List[Path]:
		"""Return generated checkpoints newest first."""
		generation_marker = f"{self.checkpoint_path.stem}_run_"
		generations = list(
			self.checkpoint_path.parent.glob(
				f"{self.checkpoint_path.stem}_run_*{self.checkpoint_path.suffix}"
			)
		)

		def generation_key(path: Path) -> Tuple[int, int]:
			generation = path.stem.removeprefix(generation_marker).split("_", 1)[0]
			try:
				episode = int(generation)
			except ValueError:
				episode = -1
			return episode, path.stat().st_mtime_ns

		return sorted(generations, key=generation_key, reverse=True)

	def load_checkpoint(self) -> bool:
		"""Load persisted training state, returning false only when no checkpoint exists."""
		generations = self._checkpoint_generations()
		source = generations[0] if generations else self.checkpoint_path
		if not source.exists():
			return False
		checkpoint = torch.load(
			source,
			map_location=self.device,
			weights_only=True,
		)
		self.policy.load_state_dict(checkpoint["policy"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		self.episodes_trained = int(checkpoint["episodes_trained"])
		return True


class SharedPolicyTrainer:
	"""Train one shared policy immediately from completed parallel episodes."""

	CHECKPOINT_VERSION = 2
	PPO_CLIP_RANGE = 0.2

	def __init__(
		self,
		observation_size: int,
		checkpoint_path: Path | str = Path(__file__).with_name(
			"car_ai_shared_checkpoint_v2.pt"
		),
		device: torch.device | str = "cuda",
		learning_rate: float = 3e-4,
		minimum_learning_rate: float = 1e-5,
		learning_rate_decay: float = 0.995,
		gamma: float = 0.99,
	) -> None:
		if learning_rate <= 0 or minimum_learning_rate <= 0:
			raise ValueError("learning rates must be positive")
		if minimum_learning_rate > learning_rate:
			raise ValueError("minimum_learning_rate cannot exceed learning_rate")
		if not 0 < learning_rate_decay <= 1:
			raise ValueError("learning_rate_decay must be in (0, 1]")
		if not 0 < gamma <= 1:
			raise ValueError("gamma must be in (0, 1]")
		self.device = torch.device(device)
		if self.device.type == "cuda" and not torch.cuda.is_available():
			raise RuntimeError(
				"CUDA was requested but is unavailable. Install a CUDA-enabled "
				"PyTorch build and verify the NVIDIA driver with "
				"`python -c \"import torch; print(torch.cuda.is_available())\"`."
			)
		if self.device.type == "cuda" and self.device.index is not None:
			if self.device.index >= torch.cuda.device_count():
				raise RuntimeError(f"CUDA device {self.device} is not available")

		self.policy = DrivingPolicy(observation_size).to(self.device)
		self.optimizer = torch.optim.AdamW(
			self.policy.parameters(),
			lr=learning_rate,
			weight_decay=1e-4,
		)
		self.minimum_learning_rate = minimum_learning_rate
		self.learning_rate = learning_rate
		self.learning_rate_decay = learning_rate_decay
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(
			self.optimizer,
			lr_lambda=lambda update: max(
				self.minimum_learning_rate / self.learning_rate,
				self.learning_rate_decay ** update,
			),
		)
		self.checkpoint_path = Path(checkpoint_path)
		self.gamma = gamma
		self.updates_completed = 0
		self.episodes_trained = 0
		self.episodes_since_batch_equivalent = 0
		self.last_ppo_metrics = {"clip_fraction": 0.0, "approx_kl": 0.0}
		self.load_checkpoint()

	def sample_actions(
		self,
		observations: Sequence[Observation],
	) -> List[Tuple[int, float]]:
		"""Sample a batch of actions without retaining autograd graphs."""
		if not observations:
			return []
		states = torch.tensor(observations, dtype=torch.float32, device=self.device)
		with torch.no_grad():
			distribution = Categorical(logits=self.policy(states))
			actions = distribution.sample()
			log_probabilities = distribution.log_prob(actions)
		return [
			(int(action), float(log_probability))
			for action, log_probability in zip(
				actions.cpu().tolist(),
				log_probabilities.cpu().tolist(),
			)
		]

	def train_batch(
		self,
		episodes: Sequence[EpisodeRollout],
		minibatch_size: int = 4096,
		optimization_epochs: int = 4,
	) -> float:
		"""Apply clipped PPO updates to complete episodes in bounded minibatches."""
		loss = self._optimize_episodes(
			episodes,
			minibatch_size=minibatch_size,
			optimization_epochs=optimization_epochs,
		)
		self.scheduler.step()
		self.episodes_trained += len(episodes)
		self.episodes_since_batch_equivalent = 0
		self.updates_completed += 1
		self.save_checkpoint()
		return loss

	def train_episode(
		self,
		episode: EpisodeRollout,
		batch_equivalent_size: int,
		minibatch_size: int = 4096,
		optimization_epochs: int = 4,
	) -> float:
		"""Update immediately from one completed run; schedule/save per equivalent."""
		if batch_equivalent_size <= 0:
			raise ValueError("batch_equivalent_size must be positive")
		loss = self._optimize_episodes(
			[episode],
			minibatch_size=minibatch_size,
			optimization_epochs=optimization_epochs,
		)
		self.episodes_trained += 1
		self.episodes_since_batch_equivalent += 1
		if self.episodes_since_batch_equivalent >= batch_equivalent_size:
			self.episodes_since_batch_equivalent -= batch_equivalent_size
			self.scheduler.step()
			self.updates_completed += 1
			self.save_checkpoint()
		return loss

	def _optimize_episodes(
		self,
		episodes: Sequence[EpisodeRollout],
		minibatch_size: int,
		optimization_epochs: int,
	) -> float:
		"""Optimize PPO policy weights without advancing checkpoint cadence."""
		if minibatch_size <= 0 or optimization_epochs <= 0:
			raise ValueError("minibatch_size and optimization_epochs must be positive")
		if not episodes:
			raise ValueError("at least one complete episode is required")

		states: List[Tensor] = []
		actions: List[int] = []
		old_log_probabilities: List[float] = []
		returns: List[float] = []
		for episode in episodes:
			if not len(episode):
				raise ValueError("episodes must contain at least one transition")
			if episode.observation_size != self.policy.layers[0].in_features:
				raise ValueError("observation size does not match the policy")
			if (
				len(episode.observations) != len(episode) * episode.observation_size
				or len(episode.rewards) != len(episode)
				or len(episode.log_probabilities) != len(episode)
			):
				raise ValueError("episode rollout buffers are misaligned")
			episode_rewards = torch.frombuffer(episode.rewards, dtype=torch.float32).tolist()
			discounted_return = 0.0
			episode_returns: List[float] = []
			for reward in reversed(episode_rewards):
				discounted_return = reward + self.gamma * discounted_return
				episode_returns.append(discounted_return)
			episode_returns.reverse()
			states.append(
				torch.frombuffer(episode.observations, dtype=torch.float32).reshape(
					len(episode), episode.observation_size
				)
			)
			actions.extend(episode.actions)
			old_log_probabilities.extend(episode.log_probabilities)
			returns.extend(episode_returns)

		return_tensor = torch.tensor(returns, dtype=torch.float32)
		return_tensor = (return_tensor - return_tensor.mean()) / (
			return_tensor.std(unbiased=False) + 1e-6
		)
		state_tensor = torch.cat(states)
		action_tensor = torch.tensor(actions, dtype=torch.long)
		old_log_probability_tensor = torch.tensor(
			old_log_probabilities, dtype=torch.float32
		)
		sample_count = len(actions)
		loss_total = 0.0
		minibatch_count = 0
		clip_fraction_total = 0.0
		approx_kl_total = 0.0
		metric_sample_count = 0

		for _ in range(optimization_epochs):
			permutation = torch.randperm(sample_count)
			for start in range(0, sample_count, minibatch_size):
				indices = permutation[start : start + minibatch_size]
				batch_states = state_tensor[indices].to(self.device)
				batch_actions = action_tensor[indices].to(self.device)
				batch_returns = return_tensor[indices].to(self.device)
				distribution = Categorical(logits=self.policy(batch_states))
				log_probabilities = distribution.log_prob(batch_actions)
				log_ratios = (
					log_probabilities
					- old_log_probability_tensor[indices].to(self.device)
				)
				ratios = torch.exp(log_ratios)
				clipped_ratios = torch.clamp(
					ratios,
					1.0 - self.PPO_CLIP_RANGE,
					1.0 + self.PPO_CLIP_RANGE,
				)
				batch_sample_count = int(indices.numel())
				clip_fraction_total += float(
					(torch.abs(ratios - 1.0) > self.PPO_CLIP_RANGE)
					.float()
					.sum()
					.detach()
					.cpu()
				)
				approx_kl_total += float(
					((ratios - 1.0) - log_ratios).sum().detach().cpu()
				)
				metric_sample_count += batch_sample_count
				policy_loss = -torch.minimum(
					ratios * batch_returns,
					clipped_ratios * batch_returns,
				).mean()
				loss = policy_loss - 0.01 * distribution.entropy().mean()

				self.optimizer.zero_grad(set_to_none=True)
				loss.backward()
				nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
				self.optimizer.step()
				loss_total += float(loss.detach().cpu())
				minibatch_count += 1

		self.last_ppo_metrics = {
			"clip_fraction": clip_fraction_total / metric_sample_count,
			"approx_kl": approx_kl_total / metric_sample_count,
		}
		return loss_total / minibatch_count

	def save_checkpoint(self) -> None:
		"""Persist training state as a generation to avoid replacing locked files."""
		self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"version": self.CHECKPOINT_VERSION,
			"policy": self.policy.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"scheduler": self.scheduler.state_dict(),
			"updates_completed": self.updates_completed,
			"episodes_trained": self.episodes_trained,
			"episodes_since_batch_equivalent": self.episodes_since_batch_equivalent,
			"learning_rate": self.learning_rate,
			"minimum_learning_rate": self.minimum_learning_rate,
			"learning_rate_decay": self.learning_rate_decay,
			"gamma": self.gamma,
		}
		generation_path = self.checkpoint_path.with_name(
			f"{self.checkpoint_path.stem}_run_{self.updates_completed:06d}_"
			f"{uuid.uuid4().hex[:8]}{self.checkpoint_path.suffix}"
		)
		temporary_path: Path | None = None
		try:
			with tempfile.NamedTemporaryFile(
				prefix=f"{self.checkpoint_path.name}.",
				suffix=".tmp",
				dir=self.checkpoint_path.parent,
				delete=False,
			) as temporary_file:
				temporary_path = Path(temporary_file.name)
			torch.save(payload, temporary_path)
			temporary_path.replace(generation_path)
		finally:
			if temporary_path is not None:
				try:
					temporary_path.unlink(missing_ok=True)
				except OSError as exc:
					warnings.warn(
						f"Could not remove temporary checkpoint {temporary_path}: {exc}",
						RuntimeWarning,
						stacklevel=2,
					)

		for old_checkpoint in self._checkpoint_generations()[5:]:
			try:
				old_checkpoint.unlink()
			except OSError as exc:
				warnings.warn(
					f"Could not remove old checkpoint {old_checkpoint}: {exc}",
					RuntimeWarning,
					stacklevel=2,
				)

	def _checkpoint_generations(self) -> List[Path]:
		generation_pattern = (
			f"{self.checkpoint_path.stem}_run_*{self.checkpoint_path.suffix}"
		)
		generations = list(self.checkpoint_path.parent.glob(generation_pattern))
		prefix = f"{self.checkpoint_path.stem}_run_"

		def generation_key(path: Path) -> Tuple[int, int]:
			try:
				update = int(path.stem.removeprefix(prefix).split("_", 1)[0])
			except ValueError:
				update = -1
			return update, path.stat().st_mtime_ns

		return sorted(generations, key=generation_key, reverse=True)

	def load_checkpoint(self) -> bool:
		"""Restore shared training state when its versioned checkpoint exists."""
		generations = self._checkpoint_generations()
		source = generations[0] if generations else self.checkpoint_path
		if not source.exists():
			return False
		checkpoint = torch.load(
			source,
			map_location=self.device,
			weights_only=True,
		)
		if checkpoint.get("version") != self.CHECKPOINT_VERSION:
			raise RuntimeError(
				f"Unsupported shared checkpoint version in {self.checkpoint_path}"
			)
		self.learning_rate = float(checkpoint["learning_rate"])
		self.minimum_learning_rate = float(checkpoint["minimum_learning_rate"])
		self.learning_rate_decay = float(checkpoint["learning_rate_decay"])
		self.gamma = float(checkpoint["gamma"])
		self.policy.load_state_dict(checkpoint["policy"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		self.scheduler.load_state_dict(checkpoint["scheduler"])
		self.updates_completed = int(checkpoint["updates_completed"])
		self.episodes_trained = int(checkpoint["episodes_trained"])
		self.episodes_since_batch_equivalent = int(
			checkpoint["episodes_since_batch_equivalent"]
		)
		return True
