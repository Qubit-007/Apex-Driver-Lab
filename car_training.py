"""Batched headless training for the procedural driving simulation."""

from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import statistics
import time
import torch

from car import ProceduralTrack
from car_ai import EpisodeRollout, SharedPolicyTrainer
from car_simulation import DrivingSimulation, Observation


@dataclass
class _EpisodeBuffer:
	"""One environment's current episode, including the version for every action."""

	rollout: EpisodeRollout
	policy_versions: list[int] = field(default_factory=list)


@dataclass
class _FinishedEpisode:
	"""Complete episode metadata retained until it is trained or dropped."""

	rollout: EpisodeRollout
	policy_versions: list[int]
	discounted_returns: list[float]
	terminal_observation: Observation
	crash_reason: str = "unknown"
	final_distance: float = 0.0

	@property
	def transitions(self) -> int:
		return len(self.rollout)

	@property
	def mean_return(self) -> float:
		return self.discounted_returns[0]

	@property
	def length(self) -> int:
		return self.transitions

	def policy_lag(self, current_version: int) -> int:
		return max(
			(current_version - version for version in self.policy_versions),
			default=0,
		)


def train_headless(
	environments: int = 1000,
	updates: int | None = 1000,
	device: str = "cuda",
	minibatch_size: int = 4096,
	optimization_epochs: int = 4,
	learning_rate_decay: float = 0.995,
	checkpoint_path: Path | str | None = None,
	seed: int = 17,
	report_every: int = 1000,
	async_reset: bool = False,
	async_episodes: int = 32,
	async_transitions: int = 8192,
	max_policy_lag: int = 4,
	min_batch_episodes: int = 128,
	max_episode_steps: int = DrivingSimulation.MAX_EPISODE_STEPS,
	obs_v2: bool = False,
	hidden_sizes: tuple[int, ...] = (256, 256),
	traffic_density: int = 14,
	traffic_speed_kmh: float = 20.0,
	ai_min_speed_kmh: float = 40.0,
	curriculum: bool = False,
) -> SharedPolicyTrainer:
	"""Run concurrent simulations and update the shared policy on each finish."""
	if environments <= 0:
		raise ValueError("environments must be positive")
	if updates is not None and updates <= 0:
		raise ValueError("updates must be positive")
	if report_every <= 0:
		raise ValueError("report_every must be positive")
	if async_episodes < 2:
		raise ValueError("async_episodes must be at least 2 to avoid single-episode updates")
	if async_transitions <= 0:
		raise ValueError("async_transitions must be positive")
	if min_batch_episodes < 2:
		raise ValueError("min_batch_episodes must be at least 2")
	if max_policy_lag < 0:
		raise ValueError("max_policy_lag cannot be negative")
	if max_episode_steps <= 0:
		raise ValueError("max_episode_steps must be positive")
	if traffic_density < 0:
		raise ValueError("traffic_density cannot be negative")
	if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
		raise ValueError("hidden_sizes must contain positive layer sizes")

	checkpoint = (
		Path(checkpoint_path)
		if checkpoint_path is not None
<<<<<<< HEAD
		else _default_checkpoint_path(
			async_reset=async_reset,
			obs_v2=obs_v2,
			hidden_sizes=hidden_sizes,
		)
=======
		else Path(__file__).with_name("car_ai_shared_checkpoint_v2.pt")
>>>>>>> 3530412f0987bce6e0e4e8e6cfe4f2121a1ee46a
	)
	trainer = SharedPolicyTrainer(
		(
			DrivingSimulation.OBSERVATION_V2_SIZE
			if obs_v2
			else DrivingSimulation.OBSERVATION_SIZE
		),
		checkpoint_path=checkpoint,
		device=device,
		learning_rate_decay=learning_rate_decay,
		hidden_sizes=hidden_sizes,
		sensible_init=obs_v2 or hidden_sizes != (256, 256),
	)
	if trainer.device.type == "cuda":
		print(f"Training device: {torch.cuda.get_device_name(trainer.device)}")
	else:
		print(f"Training device: {trainer.device}")

	simulations = []
	for index in range(environments):
		environment_seed = seed + index
		track = ProceduralTrack(
			point_spacing=1.0,
			chunks_ahead=5,
			seed=environment_seed,
		)
		simulation = DrivingSimulation(
			track=track,
			traffic_count=traffic_density,
			seed=environment_seed,
			obs_v2=obs_v2,
			traffic_speed_kmh=traffic_speed_kmh,
			ai_min_speed_kmh=ai_min_speed_kmh,
			randomize_starts=obs_v2,
			curriculum=curriculum,
		)
		# Configure the existing simulator limit from the trainer without
		# changing simulator behavior or its default.
		simulation.MAX_EPISODE_STEPS = max_episode_steps
		simulations.append(simulation)

	if async_reset:
		return _train_auto_reset(
			trainer,
			simulations,
			updates=updates,
			minibatch_size=minibatch_size,
			optimization_epochs=optimization_epochs,
			episodes_per_update=async_episodes,
			transitions_per_update=async_transitions,
			max_policy_lag=max_policy_lag,
			min_batch_episodes=min_batch_episodes,
			report_every=report_every,
			curriculum=curriculum,
		)

<<<<<<< HEAD
	observation_size = (
		DrivingSimulation.OBSERVATION_V2_SIZE
		if obs_v2
		else DrivingSimulation.OBSERVATION_SIZE
	)
	rollouts = [
		EpisodeRollout(observation_size)
		for _ in simulations
	]
	starting_update = trainer.updates_completed
	target_update = starting_update + updates
	report_count = 0
	report_distance = 0.0
	report_loss = 0.0
	report_reasons: dict[str, int] = {}
	recent_distances: deque[float] = deque(maxlen=100)
=======
	local_update = 1
	while updates is None or local_update <= updates:
		rollouts = [
			EpisodeRollout(DrivingSimulation.OBSERVATION_SIZE)
			for _ in simulations
		]
		active = list(range(environments))
		while active:
			observations = [simulations[index].observe() for index in active]
			actions = trainer.sample_actions(observations)
			next_active = []
			for index, observation, (action, log_probability) in zip(
				active, observations, actions
			):
				result = simulations[index].step(action, dt=1.0 / 30.0)
				rollouts[index].append(
					observation, action, result.reward, log_probability
				)
				if not result.done:
					next_active.append(index)
			active = next_active
>>>>>>> 3530412f0987bce6e0e4e8e6cfe4f2121a1ee46a

	while trainer.updates_completed < target_update:
		observations = [simulation.observe() for simulation in simulations]
		actions = trainer.sample_actions(observations)
		for index, observation, (action, log_probability) in zip(
			range(environments), observations, actions
		):
			if trainer.updates_completed >= target_update:
				break
			simulation = simulations[index]
			result = simulation.step(action, dt=1.0 / 30.0)
			rollouts[index].append(
				observation, action, result.reward, log_probability
			)
			if not result.done:
				continue

			loss = trainer.train_episode(
				rollouts[index],
				batch_equivalent_size=environments,
				minibatch_size=minibatch_size,
				optimization_epochs=optimization_epochs,
			)
			report_count += 1
			report_distance += simulation.player.distance
			recent_distances.append(simulation.player.distance)
			if curriculum:
				mean_distance = sum(recent_distances) / len(recent_distances)
				for environment in simulations:
					environment.set_curriculum_distance(mean_distance)
			report_loss += loss
			reason = result.crash_reason or "unknown"
			report_reasons[reason] = report_reasons.get(reason, 0) + 1

			simulation.reset()
			rollouts[index] = EpisodeRollout(observation_size)

			if report_count >= report_every or trainer.updates_completed >= target_update:
				reason_summary = ", ".join(
					f"{name}: {count}"
					for name, count in sorted(report_reasons.items())
				)
				print(
					f"Completed {trainer.episodes_trained} episodes | "
					f"updates {trainer.updates_completed}/"
					f"{target_update} batch-equivalents | "
					f"last {report_count} mean distance "
					f"{report_distance / report_count:.1f} m | "
					f"mean loss {report_loss / report_count:.4f} | "
					f"lr {trainer.optimizer.param_groups[0]['lr']:.2e} | "
					f"{reason_summary}"
				)
				report_count = 0
				report_distance = 0.0
				report_loss = 0.0
				report_reasons.clear()

	return trainer


def _finish_episode(
	buffer: _EpisodeBuffer,
	terminal_observation: Observation,
	gamma: float,
	crash_reason: str = "unknown",
	final_distance: float = 0.0,
) -> _FinishedEpisode:
	"""Finalize Monte Carlo returns without mixing data across resets."""
	discounted_returns: list[float] = []
	return_value = 0.0
	for reward in reversed(buffer.rollout.rewards):
		return_value = reward + gamma * return_value
		discounted_returns.append(return_value)
	discounted_returns.reverse()
	return _FinishedEpisode(
		rollout=buffer.rollout,
		policy_versions=buffer.policy_versions,
		discounted_returns=discounted_returns,
		terminal_observation=terminal_observation,
		crash_reason=crash_reason,
		final_distance=final_distance,
	)


def _default_checkpoint_path(
	async_reset: bool,
	obs_v2: bool,
	hidden_sizes: tuple[int, ...],
) -> Path:
	if not obs_v2 and hidden_sizes == (256, 256):
		return Path(__file__).with_name(
			"car_ai_async_checkpoint_v1.pt"
			if async_reset
			else "car_ai_shared_checkpoint_v2.pt"
		)
	model_name = "obs_v2" if obs_v2 else "obs_v1"
	hidden_name = "-".join(str(size) for size in hidden_sizes)
	collector = "async" if async_reset else "legacy"
	return Path(__file__).with_name(
		f"car_ai_{model_name}_{hidden_name}_{collector}.pt"
	)


def _filter_stale_episodes(
	episodes: list[_FinishedEpisode],
	current_policy_version: int,
	max_policy_lag: int,
) -> tuple[list[_FinishedEpisode], int]:
	"""Drop complete episodes if any sampled action exceeds the allowed lag."""
	kept = [
		episode
		for episode in episodes
		if episode.policy_lag(current_policy_version) <= max_policy_lag
	]
	return kept, len(episodes) - len(kept)


def _train_auto_reset(
	trainer: SharedPolicyTrainer,
	simulations: list[DrivingSimulation],
	updates: int,
	minibatch_size: int,
	optimization_epochs: int,
	episodes_per_update: int,
	transitions_per_update: int,
	max_policy_lag: int,
	min_batch_episodes: int,
	report_every: int,
	curriculum: bool,
) -> SharedPolicyTrainer:
	"""Collect complete episodes continuously and update only at batch thresholds."""
	observation_size = (
		getattr(
			DrivingSimulation,
			"OBSERVATION_V2_SIZE",
			DrivingSimulation.OBSERVATION_SIZE,
		)
		if getattr(simulations[0], "obs_v2", False)
		else DrivingSimulation.OBSERVATION_SIZE
	)
	environments = len(simulations)
	current_observations = [simulation.observe() for simulation in simulations]
	current_episodes = [
		_EpisodeBuffer(EpisodeRollout(observation_size))
		for _ in simulations
	]
	finished: list[_FinishedEpisode] = []
	finished_transitions = 0
	stale_dropped_since_update = 0
	starting_update = trainer.updates_completed
	target_update = starting_update + updates
	window_started = time.perf_counter()
	window_transitions = 0
	window_slot_samples = 0
	window_ticks = 0
	recent_distances: deque[float] = deque(maxlen=100)

	while trainer.updates_completed < target_update:
		# Every slot samples from the same immutable policy version for this tick.
		collection_version = trainer.updates_completed
		actions = trainer.sample_actions(current_observations)
		completed_this_tick = 0
		for index, (action, log_probability) in enumerate(actions):
			simulation = simulations[index]
			episode = current_episodes[index]
			result = simulation.step(action, dt=1.0 / 30.0)
			episode.rollout.append(
				current_observations[index],
				action,
				result.reward,
				log_probability,
			)
			episode.policy_versions.append(collection_version)
			window_transitions += 1
			window_slot_samples += 1
			current_observations[index] = result.observation

			if not result.done:
				continue

			# Keep the terminal observation before resetting, so episode-boundary
			# data remains available even though no bootstrap is used today.
			finished_episode = _finish_episode(
				episode,
				terminal_observation=result.observation,
				gamma=trainer.gamma,
				crash_reason=result.crash_reason or "unknown",
				final_distance=simulation.player.distance,
			)
			finished.append(finished_episode)
			finished_transitions += finished_episode.transitions
			completed_this_tick += 1
			recent_distances.append(simulation.player.distance)
			if curriculum:
				mean_distance = sum(recent_distances) / len(recent_distances)
				for environment in simulations:
					environment.set_curriculum_distance(mean_distance)

			# Reset only this environment on its terminal step. The returned
			# observation is the first state of its next episode.
			current_observations[index] = simulation.reset()
			current_episodes[index] = _EpisodeBuffer(
				EpisodeRollout(observation_size)
			)

		window_ticks += 1
		if not finished or not (
			len(finished) >= episodes_per_update
			or finished_transitions >= transitions_per_update
		):
			continue

		# Versions are recorded per action, but stale episodes are dropped whole
		# so Monte Carlo returns never cross a missing transition or episode edge.
		policy_version = trainer.updates_completed
		kept_episodes, stale_dropped = _filter_stale_episodes(
			finished,
			current_policy_version=policy_version,
			max_policy_lag=max_policy_lag,
		)
		stale_dropped_since_update += stale_dropped
		finished = kept_episodes
		finished_transitions = sum(
			episode.transitions for episode in finished
		)

		# Keep gathering fresh complete episodes until the configured batch
		# minimum is reached; do not update from only the few survivors.
		if len(kept_episodes) < min_batch_episodes:
			continue

		update_policy_version = policy_version
		loss = trainer.train_batch(
			[episode.rollout for episode in kept_episodes],
			minibatch_size=minibatch_size,
			optimization_epochs=optimization_epochs,
		)
<<<<<<< HEAD
		# The trainer owns one live policy shared by every environment, so the
		# next vector tick automatically uses its new weights and update version.
		policy_version = trainer.updates_completed

		now = time.perf_counter()
		elapsed = max(now - window_started, 1e-9)
		steps_per_second = window_transitions / elapsed
		active_percent = (
			100.0 * window_slot_samples / (environments * window_ticks)
			if window_ticks
			else 0.0
		)
		mean_length = sum(episode.length for episode in kept_episodes) / len(
			kept_episodes
		)
		mean_return = sum(episode.mean_return for episode in kept_episodes) / len(
			kept_episodes
		)
		mean_lag = sum(
			sum(update_policy_version - version for version in episode.policy_versions)
			/ len(episode.policy_versions)
			for episode in kept_episodes
		) / len(kept_episodes)
		distances = [episode.final_distance for episode in kept_episodes]
		mean_distance = sum(distances) / len(distances)
		median_distance = statistics.median(distances)
		reason_counts: dict[str, int] = {}
		for episode in kept_episodes:
			reason_counts[episode.crash_reason] = (
				reason_counts.get(episode.crash_reason, 0) + 1
			)
		reason_summary = ", ".join(
			f"{reason}={count}"
			for reason, count in sorted(reason_counts.items())
		)
		ppo_metrics = trainer.last_ppo_metrics
		print(
			f"Update {trainer.updates_completed}/{target_update} | "
			f"{window_transitions} steps @ {steps_per_second:.0f} steps/s | "
			f"envs active {active_percent:.1f}% | "
			f"episodes used {len(kept_episodes)} | "
			f"episodes dropped {stale_dropped_since_update} | "
			f"mean episode length {mean_length:.1f} | "
			f"distance mean/median {mean_distance:.1f}/{median_distance:.1f} m | "
			f"mean return {mean_return:.3f} | "
			f"policy lag {mean_lag:.2f} updates | "
			f"end reasons [{reason_summary}] | "
			f"PPO clip fraction {ppo_metrics['clip_fraction']:.3f} | "
			f"approx KL {ppo_metrics['approx_kl']:.5f} | "
			f"loss {loss:.4f}"
		)
		finished = []
		finished_transitions = 0
		stale_dropped_since_update = 0
		window_started = now
		window_transitions = 0
		window_slot_samples = 0
		window_ticks = 0
=======
		if (
			local_update % report_every == 0
			or (updates is not None and local_update == updates)
		):
			mean_distance = sum(
				simulation.player.distance for simulation in simulations
			) / environments
			reason_counts: dict[str, int] = {}
			for simulation in simulations:
				reason = simulation.crash_reason or "unknown"
				reason_counts[reason] = reason_counts.get(reason, 0) + 1
			reason_summary = ", ".join(
				f"{reason}: {count}" for reason, count in sorted(reason_counts.items())
			)
			learning_rate = trainer.optimizer.param_groups[0]["lr"]
			print(
				f"Update {trainer.updates_completed} "
				f"({local_update}/{updates or 'until stopped'}) | "
				f"{environments} episodes | "
				f"mean distance {mean_distance:.1f} m | "
				f"loss {loss:.4f} | lr {learning_rate:.2e} | "
				f"{reason_summary}"
			)

		for simulation in simulations:
			simulation.reset()
		local_update += 1
>>>>>>> 3530412f0987bce6e0e4e8e6cfe4f2121a1ee46a

	return trainer
