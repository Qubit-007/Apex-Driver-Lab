"""Small online policy-gradient driver for the procedural car game."""

from pathlib import Path
import tempfile
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from car_simulation import Observation


class DrivingPolicy(nn.Module):
	"""Map normalized track/traffic observations to nine driving actions."""

	def __init__(self, observation_size: int, action_count: int = 9) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(observation_size, 128),
			nn.Tanh(),
			nn.Linear(128, 128),
			nn.Tanh(),
			nn.Linear(128, action_count),
		)

	def forward(self, observation: Tensor) -> Tensor:
		return self.layers(observation)


class OnlineDriver:
	"""Collect complete runs and update a stochastic policy using REINFORCE."""

	def __init__(
		self,
		observation_size: int,
		checkpoint_path: Path | str = Path(__file__).with_name("car_ai_checkpoint_v2.pt"),
		device: torch.device | str | None = None,
		learning_rate: float = 3e-4,
		gamma: float = 0.99,
	) -> None:
		self.device = torch.device(
			device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
		)
		self.policy = DrivingPolicy(observation_size).to(self.device)
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
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
		"""Atomically save model and optimizer state beside the game source."""
		self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"policy": self.policy.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"episodes_trained": self.episodes_trained,
		}
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
			temporary_path.replace(self.checkpoint_path)
		finally:
			if temporary_path is not None:
				temporary_path.unlink(missing_ok=True)

	def load_checkpoint(self) -> bool:
		"""Load persisted training state, returning false only when no checkpoint exists."""
		if not self.checkpoint_path.exists():
			return False
		checkpoint = torch.load(
			self.checkpoint_path,
			map_location=self.device,
			weights_only=True,
		)
		self.policy.load_state_dict(checkpoint["policy"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		self.episodes_trained = int(checkpoint["episodes_trained"])
		return True
