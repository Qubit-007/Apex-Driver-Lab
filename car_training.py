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
	report_every: int = 1000,
) -> SharedPolicyTrainer:
	"""Run concurrent simulations and update the shared policy on each finish."""
	if environments <= 0:
		raise ValueError("environments must be positive")
	if updates <= 0:
		raise ValueError("updates must be positive")
	if report_every <= 0:
		raise ValueError("report_every must be positive")

	checkpoint = (
		Path(checkpoint_path)
		if checkpoint_path is not None
		else Path(__file__).with_name("car_ai_shared_checkpoint_v2.pt")
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

	rollouts = [
		EpisodeRollout(DrivingSimulation.OBSERVATION_SIZE)
		for _ in simulations
	]
	starting_update = trainer.updates_completed
	target_update = starting_update + updates
	report_count = 0
	report_distance = 0.0
	report_loss = 0.0
	report_reasons: dict[str, int] = {}

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
			report_loss += loss
			reason = result.crash_reason or "unknown"
			report_reasons[reason] = report_reasons.get(reason, 0) + 1

			simulation.reset()
			rollouts[index] = EpisodeRollout(DrivingSimulation.OBSERVATION_SIZE)

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
