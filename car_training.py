"""Batched headless training for the procedural driving simulation."""

from pathlib import Path
import torch

from car import ProceduralTrack
from car_ai import EpisodeRollout, SharedPolicyTrainer
from car_simulation import DrivingSimulation


def train_headless(
	environments: int = 1000,
	updates: int = 1000,
	device: str = "cuda",
	minibatch_size: int = 4096,
	optimization_epochs: int = 4,
	learning_rate_decay: float = 0.995,
	checkpoint_path: Path | str | None = None,
	seed: int = 17,
	report_every: int = 1,
) -> SharedPolicyTrainer:
	"""Advance ``environments`` in lockstep and update after each shared batch."""
	if environments <= 0:
		raise ValueError("environments must be positive")
	if updates <= 0:
		raise ValueError("updates must be positive")
	if report_every <= 0:
		raise ValueError("report_every must be positive")

	checkpoint = (
		Path(checkpoint_path)
		if checkpoint_path is not None
		else Path(__file__).with_name("car_ai_shared_checkpoint_v1.pt")
	)
	trainer = SharedPolicyTrainer(
		DrivingSimulation.OBSERVATION_SIZE,
		checkpoint_path=checkpoint,
		device=device,
		learning_rate_decay=learning_rate_decay,
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
		simulations.append(
			DrivingSimulation(track=track, traffic_count=14, seed=environment_seed)
		)

	for local_update in range(1, updates + 1):
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

		loss = trainer.train_batch(
			rollouts,
			minibatch_size=minibatch_size,
			optimization_epochs=optimization_epochs,
		)
		if local_update % report_every == 0 or local_update == updates:
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
				f"({local_update}/{updates}) | "
				f"{environments} episodes | "
				f"mean distance {mean_distance:.1f} m | "
				f"loss {loss:.4f} | lr {learning_rate:.2e} | "
				f"{reason_summary}"
			)

		for simulation in simulations:
			simulation.reset()

	return trainer
