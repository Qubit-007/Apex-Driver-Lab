"""Focused tests for the headless procedural driving game."""

import itertools
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from car_ai import EpisodeRollout, OnlineDriver, SharedPolicyTrainer
from car_simulation import DrivingSimulation, TrafficCar


class DrivingSimulationTests(unittest.TestCase):
	def test_observation_has_fixed_normalized_shape_and_traffic_count(self) -> None:
		simulation = DrivingSimulation(traffic_count=14, seed=2)
		observation = simulation.observe()

		self.assertEqual(len(observation), simulation.OBSERVATION_SIZE)
		self.assertEqual(len(simulation.traffic), 14)
		self.assertTrue(all(-1.0 <= value <= 1.0 for value in observation))
		self.assertAlmostEqual(
			observation[simulation.LEFT_CLEARANCE_INDEX],
			(simulation.track.track_width / 2 - simulation.PLAYER_HALF_WIDTH)
			/ simulation.track.track_width,
		)
		self.assertAlmostEqual(
			observation[simulation.RIGHT_CLEARANCE_INDEX],
			observation[simulation.LEFT_CLEARANCE_INDEX],
		)
		self.assertTrue(
			all(
				car.speed == simulation.TRAFFIC_SPEED
				for car in simulation.traffic
			)
		)
		self.assertEqual(simulation.TRAFFIC_SPEED * 3.6, 20.0)

	def test_observation_reports_both_road_clearances_and_nearest_traffic(self) -> None:
		simulation = DrivingSimulation(traffic_count=2, seed=2)
		simulation.player.lateral = 2.0
		simulation.player.distance = 20.0
		simulation.traffic[0].distance = 30.0
		simulation.traffic[0].lateral = 3.0
		simulation.traffic[1].distance = 12.0
		simulation.traffic[1].lateral = -2.0

		observation = simulation.observe()
		left_index = simulation.LEFT_CLEARANCE_INDEX
		right_index = simulation.RIGHT_CLEARANCE_INDEX
		traffic_index = simulation.TRAFFIC_FEATURES_START_INDEX

		self.assertAlmostEqual(
			observation[left_index],
			(simulation.track.track_width / 2 + 2.0 - simulation.PLAYER_HALF_WIDTH)
			/ simulation.track.track_width,
		)
		self.assertAlmostEqual(
			observation[right_index],
			(simulation.track.track_width / 2 - 2.0 - simulation.PLAYER_HALF_WIDTH)
			/ simulation.track.track_width,
		)
		self.assertAlmostEqual(
			observation[traffic_index],
			(
				simulation.track.center_x_at(12.0) - 2.0
				- simulation.track.center_x_at(20.0) - 2.0
			) / simulation.HORIZONTAL_DISTANCE_SCALE,
		)
		self.assertAlmostEqual(
			observation[traffic_index + 1],
			(12.0 - 20.0) / simulation.VERTICAL_DISTANCE_SCALE,
		)
		self.assertAlmostEqual(
			observation[traffic_index + 2],
			(-2.0 - 2.0) / simulation.LATERAL_DISTANCE_SCALE,
		)
		self.assertAlmostEqual(
			observation[traffic_index + 3],
			(simulation.TRAFFIC_SPEED - simulation.player.speed)
			/ simulation.RELATIVE_SPEED_SCALE,
		)

		simulation.traffic[1].distance = 40.0
		simulation.traffic[1].lateral = 5.0
		ahead_observation = simulation.observe()
		self.assertGreater(ahead_observation[traffic_index + 1], 0.0)
		self.assertLess(ahead_observation[traffic_index], 1.0)
		self.assertEqual(
			ahead_observation[traffic_index + 1],
			(simulation.traffic[0].distance - simulation.player.distance)
			/ simulation.VERTICAL_DISTANCE_SCALE,
		)
		self.assertAlmostEqual(
			ahead_observation[traffic_index + 2],
			(simulation.traffic[0].lateral - simulation.player.lateral)
			/ simulation.LATERAL_DISTANCE_SCALE,
		)

	def test_player_brakes_but_never_goes_below_40_kmh(self) -> None:
		simulation = DrivingSimulation(traffic_count=0, seed=5)
		self.assertGreaterEqual(
			simulation.player.speed,
			simulation.PLAYER_MIN_SPEED,
		)
		self.assertEqual(simulation.PLAYER_MIN_SPEED * 3.6, 40.0)

		for _ in range(20):
			result = simulation.step(action=3, dt=0.5)
			if result.done:
				simulation.reset()

		self.assertEqual(simulation.player.speed, simulation.PLAYER_MIN_SPEED)
		self.assertGreaterEqual(simulation.player.speed * 3.6, 40.0)

	def test_traffic_collision_ends_episode_and_reset_restarts_player(self) -> None:
		simulation = DrivingSimulation(traffic_count=1, seed=1)
		simulation.traffic = [
			TrafficCar(distance=0.0, lateral=0.0, speed=18.0, car_id=100)
		]

		result = simulation.step(action=4, dt=0.1)

		self.assertTrue(result.done)
		self.assertEqual(result.crash_reason, "traffic collision")
		self.assertLess(result.reward, -30.0)
		with self.assertRaises(RuntimeError):
			simulation.step(action=4, dt=0.1)

		observation = simulation.reset()
		self.assertEqual(simulation.player.distance, 0.0)
		self.assertEqual(simulation.player.lateral, 0.0)
		self.assertEqual(simulation.steps, 0)
		self.assertEqual(simulation.crash_reason, None)
		self.assertEqual(len(observation), simulation.OBSERVATION_SIZE)

	def test_leaving_road_ends_episode(self) -> None:
		simulation = DrivingSimulation(traffic_count=0, seed=3)
		simulation.player.lateral = 5.0

		result = simulation.step(action=4, dt=0.05)

		self.assertTrue(result.done)
		self.assertEqual(result.crash_reason, "off the road")
		self.assertLess(result.reward, -20.0)

	def test_traffic_is_replenished_ahead_as_player_moves(self) -> None:
		simulation = DrivingSimulation(traffic_count=2, seed=4)
		simulation.traffic[0].distance = -100.0
		simulation.player.distance = 60.0

		simulation._update_traffic(0.1)

		self.assertEqual(len(simulation.traffic), 2)
		self.assertTrue(
			any(car.distance > simulation.player.distance + 80.0 for car in simulation.traffic)
		)


class OnlineDriverTests(unittest.TestCase):
	def test_policy_has_doubled_hidden_layers_and_uses_adamw(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			driver = OnlineDriver(
				DrivingSimulation.OBSERVATION_SIZE,
				Path(directory) / "fresh.pt",
			)

			self.assertEqual(driver.policy.layers[0].out_features, 256)
			self.assertEqual(driver.policy.layers[2].out_features, 256)
			self.assertIsInstance(driver.optimizer, torch.optim.AdamW)
			self.assertEqual(
				driver.checkpoint_path.name,
				"fresh.pt",
			)

	def test_policy_update_changes_weights_and_checkpoint_reloads(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			checkpoint = Path(directory) / "driver.pt"
			driver = OnlineDriver(DrivingSimulation.OBSERVATION_SIZE, checkpoint)
			initial = [parameter.detach().clone() for parameter in driver.policy.parameters()]
			action, _ = driver.choose_action(
				tuple(0.1 for _ in range(DrivingSimulation.OBSERVATION_SIZE))
			)
			self.assertIn(action, range(9))
			for fill_value, chosen_action, reward in ((-0.5, 0, 0.0), (0.5, 8, 1.0)):
				state = torch.full(
					(DrivingSimulation.OBSERVATION_SIZE,),
					fill_value,
					device=driver.device,
				)
				logits = driver.policy(state)
				log_probability = Categorical(logits=logits).log_prob(
					torch.tensor(chosen_action, device=driver.device)
				)
				driver.record(log_probability, reward)
			loss = driver.finish_episode()

			self.assertIsNotNone(loss)
			generations = list(checkpoint.parent.glob("driver_run_*.pt"))
			self.assertEqual(len(generations), 1)
			self.assertEqual(driver.episodes_trained, 1)
			self.assertTrue(
				any(
					not torch.equal(before, after)
					for before, after in zip(initial, driver.policy.parameters())
				)
			)

			reloaded = OnlineDriver(DrivingSimulation.OBSERVATION_SIZE, checkpoint)
			self.assertEqual(reloaded.episodes_trained, 1)
			for expected, actual in zip(driver.policy.parameters(), reloaded.policy.parameters()):
				self.assertTrue(torch.equal(expected, actual))

	def test_checkpoint_save_avoids_replacing_locked_base_file(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			checkpoint = Path(directory) / "driver.pt"
			driver = OnlineDriver(DrivingSimulation.OBSERVATION_SIZE, checkpoint)
			torch.save(
				{
					"policy": driver.policy.state_dict(),
					"optimizer": driver.optimizer.state_dict(),
					"episodes_trained": 4,
				},
				checkpoint,
			)
			original_contents = checkpoint.read_bytes()
			driver = OnlineDriver(DrivingSimulation.OBSERVATION_SIZE, checkpoint)
			for chosen_action, reward in ((0, 0.0), (8, 1.0)):
				state = torch.full(
					(DrivingSimulation.OBSERVATION_SIZE,),
					float(chosen_action),
					device=driver.device,
				)
				log_probability = Categorical(
					logits=driver.policy(state)
				).log_prob(torch.tensor(chosen_action, device=driver.device))
				driver.record(log_probability, reward)

			original_replace = Path.replace

			def reject_existing_target(source: Path, target: Path) -> Path:
				if target == checkpoint:
					raise PermissionError("simulated OneDrive lock")
				return original_replace(source, target)

			with patch.object(Path, "replace", reject_existing_target):
				driver.finish_episode()

			self.assertEqual(checkpoint.read_bytes(), original_contents)
			self.assertEqual(driver.episodes_trained, 5)
			reloaded = OnlineDriver(DrivingSimulation.OBSERVATION_SIZE, checkpoint)
			self.assertEqual(reloaded.episodes_trained, 5)
			self.assertEqual(len(list(checkpoint.parent.glob("driver_run_*.pt"))), 1)

	def test_episode_with_no_actions_does_not_update(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			driver = OnlineDriver(
				DrivingSimulation.OBSERVATION_SIZE,
				Path(directory) / "unused.pt",
			)

			self.assertIsNone(driver.finish_episode())
			self.assertEqual(driver.episodes_trained, 0)


class SharedPolicyTrainerTests(unittest.TestCase):
	def test_batched_update_decays_schedule_and_restores_checkpoint(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			checkpoint = Path(directory) / "shared.pt"
			trainer = SharedPolicyTrainer(
				DrivingSimulation.OBSERVATION_SIZE,
				checkpoint_path=checkpoint,
				device="cpu",
				learning_rate_decay=0.5,
			)
			initial_rate = trainer.optimizer.param_groups[0]["lr"]
			episodes = []
			for episode_index in range(2):
				rollout = EpisodeRollout(DrivingSimulation.OBSERVATION_SIZE)
				for step in range(3):
					observation = tuple(
						float(episode_index + step) / 10.0
						for _ in range(DrivingSimulation.OBSERVATION_SIZE)
					)
					with torch.no_grad():
						distribution = Categorical(logits=trainer.policy(
							torch.tensor(observation, dtype=torch.float32)
						))
						action = (episode_index + step) % 9
						log_probability = float(
							distribution.log_prob(torch.tensor(action))
						)
					rollout.append(observation, action, 1.0, log_probability)
				episodes.append(rollout)

			loss = trainer.train_batch(
				episodes,
				minibatch_size=3,
				optimization_epochs=1,
			)

			self.assertTrue(torch.isfinite(torch.tensor(loss)))
			self.assertGreaterEqual(
				trainer.last_ppo_metrics["clip_fraction"],
				0.0,
			)
			self.assertLessEqual(
				trainer.last_ppo_metrics["clip_fraction"],
				1.0,
			)
			self.assertTrue(
				torch.isfinite(
					torch.tensor(trainer.last_ppo_metrics["approx_kl"])
				)
			)
			self.assertEqual(trainer.updates_completed, 1)
			self.assertEqual(trainer.episodes_trained, 2)
			self.assertLess(trainer.optimizer.param_groups[0]["lr"], initial_rate)
			generations = list(checkpoint.parent.glob("shared_run_*.pt"))
			self.assertEqual(len(generations), 1)

			reloaded = SharedPolicyTrainer(
				DrivingSimulation.OBSERVATION_SIZE,
				checkpoint_path=checkpoint,
				device="cpu",
				learning_rate_decay=0.99,
			)
			self.assertEqual(reloaded.updates_completed, 1)
			self.assertEqual(reloaded.episodes_trained, 2)
			self.assertEqual(reloaded.learning_rate_decay, 0.5)
			self.assertEqual(
				reloaded.optimizer.param_groups[0]["lr"],
				trainer.optimizer.param_groups[0]["lr"],
			)
			for expected, actual in zip(
				trainer.policy.parameters(), reloaded.policy.parameters()
			):
				self.assertTrue(torch.equal(expected, actual))

	def test_action_sampling_batches_without_gradients(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			trainer = SharedPolicyTrainer(
				DrivingSimulation.OBSERVATION_SIZE,
				checkpoint_path=Path(directory) / "shared.pt",
				device="cpu",
			)

			actions = trainer.sample_actions(
				[
					tuple(0.0 for _ in range(DrivingSimulation.OBSERVATION_SIZE)),
					tuple(0.5 for _ in range(DrivingSimulation.OBSERVATION_SIZE)),
				]
			)

			self.assertEqual(len(actions), 2)
			self.assertTrue(all(action in range(9) for action, _ in actions))
			self.assertEqual(trainer.sample_actions([]), [])

	def test_cuda_request_fails_with_actionable_message_when_unavailable(self) -> None:
		with patch("car_ai.torch.cuda.is_available", return_value=False):
			with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
				SharedPolicyTrainer(
					DrivingSimulation.OBSERVATION_SIZE,
					device="cuda",
				)

	def test_single_episode_updates_immediately_and_checkpoints_per_equivalent(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			checkpoint = Path(directory) / "shared_v2.pt"
			trainer = SharedPolicyTrainer(
				DrivingSimulation.OBSERVATION_SIZE,
				checkpoint_path=checkpoint,
				device="cpu",
				learning_rate_decay=0.5,
			)
			initial_rate = trainer.optimizer.param_groups[0]["lr"]
			observation = tuple(
				0.1 for _ in range(DrivingSimulation.OBSERVATION_SIZE)
			)

			def completed_episode() -> EpisodeRollout:
				action, log_probability = trainer.sample_actions([observation])[0]
				episode = EpisodeRollout(DrivingSimulation.OBSERVATION_SIZE)
				episode.append(observation, action, 1.0, log_probability)
				return episode

			first_loss = trainer.train_episode(
				completed_episode(),
				batch_equivalent_size=2,
				minibatch_size=8,
				optimization_epochs=1,
			)
			self.assertTrue(torch.isfinite(torch.tensor(first_loss)))
			self.assertEqual(trainer.episodes_trained, 1)
			self.assertEqual(trainer.updates_completed, 0)
			self.assertEqual(trainer.episodes_since_batch_equivalent, 1)
			self.assertEqual(trainer.optimizer.param_groups[0]["lr"], initial_rate)
			self.assertEqual(list(checkpoint.parent.glob("shared_v2_run_*.pt")), [])

			second_loss = trainer.train_episode(
				completed_episode(),
				batch_equivalent_size=2,
				minibatch_size=8,
				optimization_epochs=1,
			)
			self.assertTrue(torch.isfinite(torch.tensor(second_loss)))
			self.assertEqual(trainer.episodes_trained, 2)
			self.assertEqual(trainer.updates_completed, 1)
			self.assertEqual(trainer.episodes_since_batch_equivalent, 0)
			self.assertLess(trainer.optimizer.param_groups[0]["lr"], initial_rate)
			generations = list(checkpoint.parent.glob("shared_v2_run_*.pt"))
			self.assertEqual(len(generations), 1)
			payload = torch.load(generations[0], map_location="cpu", weights_only=True)
			self.assertEqual(payload["version"], 2)

			reloaded = SharedPolicyTrainer(
				DrivingSimulation.OBSERVATION_SIZE,
				checkpoint_path=checkpoint,
				device="cpu",
			)
			self.assertEqual(reloaded.episodes_trained, 2)
			self.assertEqual(reloaded.updates_completed, 1)
			self.assertEqual(reloaded.episodes_since_batch_equivalent, 0)
			self.assertEqual(
				reloaded.optimizer.param_groups[0]["lr"],
				trainer.optimizer.param_groups[0]["lr"],
			)


class HeadlessTrainingTests(unittest.TestCase):
	def test_finished_environment_updates_and_restarts_while_others_continue(self) -> None:
		events = []
		simulations = []

		class FakeSimulation:
			OBSERVATION_SIZE = 1

			def __init__(self, **_kwargs) -> None:
				self.index = len(simulations)
				self.total_steps = 0
				self.player = types.SimpleNamespace(distance=0.0)
				self.reset_count = 0
				simulations.append(self)

			def observe(self):
				return (float(self.index),)

			def step(self, _action: int, dt: float):
				self.total_steps += 1
				self.player.distance += dt
				done = (
					(self.index == 0 and self.total_steps == 1)
					or (self.index == 1 and self.total_steps == 3)
				)
				events.append(("step", self.index, self.total_steps))
				return types.SimpleNamespace(
					reward=1.0,
					done=done,
					crash_reason="test crash" if done else None,
				)

			def reset(self) -> None:
				self.reset_count += 1
				events.append(("reset", self.index, self.reset_count))
				self.player.distance = 0.0

		class FakeTrainer:
			def __init__(self, *_args, **_kwargs) -> None:
				self.device = torch.device("cpu")
				self.episodes_trained = 0
				self.updates_completed = 0
				self.optimizer_steps = 0
				self.optimizer = types.SimpleNamespace(
					param_groups=[{"lr": 3e-4}]
				)

			def sample_actions(self, observations):
				events.append(("sample", self.optimizer_steps))
				return [(4, -0.5) for _ in observations]

			def train_episode(
				self,
				episode,
				batch_equivalent_size,
				minibatch_size,
				optimization_epochs,
			):
				self.episodes_trained += 1
				self.optimizer_steps += 1
				events.append(
					("train", len(episode), batch_equivalent_size)
				)
				if self.episodes_trained % batch_equivalent_size == 0:
					self.updates_completed += 1
				return 0.25

		from car_training import train_headless

		with patch("car_training.DrivingSimulation", FakeSimulation), patch(
			"car_training.SharedPolicyTrainer", FakeTrainer
		), patch("builtins.print"):
			trainer = train_headless(
				environments=2,
				updates=1,
				device="cpu",
				minibatch_size=8,
				optimization_epochs=1,
				report_every=100,
			)

		self.assertEqual(trainer.episodes_trained, 2)
		self.assertEqual(trainer.updates_completed, 1)
		self.assertEqual([simulation.reset_count for simulation in simulations], [1, 1])
		self.assertEqual(
			[item for item in events if item[0] == "train"],
			[("train", 1, 2), ("train", 3, 2)],
		)
		self.assertEqual(
			[item for item in events if item[0] == "sample"],
			[("sample", 0), ("sample", 1), ("sample", 1)],
		)
		self.assertLess(
			events.index(("reset", 0, 1)),
			events.index(("step", 1, 2)),
		)

	def test_async_reset_batches_complete_episodes_and_tracks_policy_versions(self) -> None:
		from car_training import (
			_EpisodeBuffer,
			_FinishedEpisode,
			_filter_stale_episodes,
			_finish_episode,
			train_headless,
		)

		events = []
		simulations = []

		class FakeSimulation:
			OBSERVATION_SIZE = 1
			MAX_EPISODE_STEPS = 1800

			def __init__(self, **_kwargs) -> None:
				self.index = len(simulations)
				self.episode_steps = 0
				self.episodes_completed = 0
				self.player = types.SimpleNamespace(distance=0.0)
				self.max_steps_at_creation = self.MAX_EPISODE_STEPS
				simulations.append(self)

			def observe(self):
				return (float(self.index),)

			def step(self, _action: int, dt: float):
				self.episode_steps += 1
				self.player.distance += 1.0
				limit = 1 if self.index == 0 else 3
				done = self.episode_steps >= limit
				events.append(("step", self.index, self.episode_steps))
				terminal = (float(100 + self.index), float(self.episode_steps))
				return types.SimpleNamespace(
					reward=float(self.episode_steps),
					done=done,
					crash_reason="episode limit" if done else None,
					observation=terminal if done else (float(self.index),),
				)

			def reset(self):
				self.episodes_completed += 1
				self.episode_steps = 0
				self.player.distance = 0.0
				events.append(("reset", self.index, self.episodes_completed))
				return (float(self.index),)

		class FakeTrainer:
			gamma = 0.5

			def __init__(self, *_args, **_kwargs) -> None:
				self.device = torch.device("cpu")
				self.updates_completed = 0
				self.episodes_trained = 0
				self.optimizer = types.SimpleNamespace(
					param_groups=[{"lr": 3e-4}]
				)
				self.batches = []
				self.sampled_versions = []
				self.last_ppo_metrics = {
					"clip_fraction": 0.125,
					"approx_kl": 0.03125,
				}

			def sample_actions(self, observations):
				version = self.updates_completed
				self.sampled_versions.append(version)
				events.append(("sample", version))
				return [(4, float(version)) for _ in observations]

			def train_batch(self, episodes, **kwargs):
				self.batches.append(episodes)
				self.episodes_trained += len(episodes)
				self.updates_completed += 1
				events.append(("train_batch", len(episodes)))
				return 0.25

		logs = []
		with patch("car_training.DrivingSimulation", FakeSimulation), patch(
			"car_training.SharedPolicyTrainer", FakeTrainer
		), patch("builtins.print", side_effect=logs.append):
			trainer = train_headless(
				environments=2,
				updates=2,
				device="cpu",
				async_reset=True,
				async_episodes=2,
				async_transitions=1000,
				max_policy_lag=4,
				min_batch_episodes=3,
				max_episode_steps=7,
			)

		self.assertEqual(trainer.updates_completed, 2)
		self.assertEqual(trainer.episodes_trained, 8)
		self.assertEqual(len(trainer.batches), 2)
		self.assertTrue(all(len(batch) >= 3 for batch in trainer.batches))
		self.assertEqual(
			[len(episode) for episode in trainer.batches[0]],
			[1, 1, 1, 3],
		)
		self.assertEqual(
			[len(episode) for episode in trainer.batches[1]],
			[1, 1, 1, 3],
		)
		self.assertTrue(
			all(
				all(value == 0.0 for value in episode.log_probabilities)
				for episode in trainer.batches[0]
			)
		)
		self.assertTrue(
			all(
				all(value == 1.0 for value in episode.log_probabilities)
				for episode in trainer.batches[1]
			)
		)
		self.assertEqual(
			[simulation.MAX_EPISODE_STEPS for simulation in simulations],
			[7, 7],
		)
		self.assertEqual([simulation.episodes_completed for simulation in simulations], [6, 2])
		self.assertIn(("step", 1, 2), events)
		self.assertLess(
			events.index(("reset", 0, 1)),
			events.index(("step", 1, 2)),
		)
		self.assertEqual(trainer.sampled_versions, [0, 0, 0, 1, 1, 1])
		update_logs = [log for log in logs if log.startswith("Update ")]
		self.assertEqual(len(update_logs), 2)
		self.assertTrue(
			all(
				"episodes used 4" in log
				and "distance mean/median 1.5/1.0 m" in log
				and "end reasons [episode limit=4]" in log
				and "PPO clip fraction 0.125" in log
				and "approx KL 0.03125" in log
				for log in update_logs
			)
		)

		rollout = EpisodeRollout(1)
		rollout.append((0.0,), 4, 1.0, -0.5)
		rollout.append((0.0,), 4, 2.0, -0.5)
		finished = _finish_episode(
			_EpisodeBuffer(rollout, [0, 0]),
			terminal_observation=(9.0,),
			gamma=0.5,
		)
		self.assertEqual(finished.discounted_returns, [2.0, 2.0])
		self.assertEqual(finished.terminal_observation, (9.0,))
		self.assertEqual(finished.policy_lag(current_version=2), 2)
		fresh = _FinishedEpisode(
			rollout=rollout,
			policy_versions=[2, 2],
			discounted_returns=[2.0, 2.0],
			terminal_observation=(9.0,),
		)
		kept, stale_count = _filter_stale_episodes(
			[finished, fresh],
			current_policy_version=2,
			max_policy_lag=1,
		)
		self.assertEqual(kept, [fresh])
		self.assertEqual(stale_count, 1)


class GameGuiTests(unittest.TestCase):
	def test_gui_runs_multiple_ai_frames_headlessly(self) -> None:
		class FakeCanvas:
			def __init__(self, *args, **kwargs) -> None:
				self.options = (args, kwargs)
				self.items = []
				self.frames = []

			def pack(self, **kwargs) -> None:
				self.pack_options = kwargs

			def winfo_width(self) -> int:
				return 800

			def winfo_height(self) -> int:
				return 600

			def delete(self, _tag: str) -> None:
				self.deleted_tag = _tag
				if self.items:
					self.frames.append(self.items.copy())
				self.items.clear()

			def create_polygon(self, *args, **kwargs) -> None:
				self.items.append(("polygon", args, kwargs))

			def create_line(self, *args, **kwargs) -> None:
				self.items.append(("line", args, kwargs))

			def create_rectangle(self, *args, **kwargs) -> None:
				self.items.append(("rectangle", args, kwargs))

			def create_text(self, *args, **kwargs) -> None:
				self.items.append(("text", args, kwargs))

		class FakeRoot:
			def __init__(self, canvas) -> None:
				self.canvas = canvas
				self.callbacks = []

			def title(self, value: str) -> None:
				self.window_title = value

			def geometry(self, value: str) -> None:
				self.window_geometry = value

			def minsize(self, *_args) -> None:
				self.minimum_size = _args

			def configure(self, **_kwargs) -> None:
				self.window_options = _kwargs

			def after(self, _delay: int, callback) -> None:
				self.last_delay = _delay
				self.callbacks.append(callback)

			def mainloop(self) -> None:
				for _ in range(8):
					self.callbacks.pop(0)()

		class FakeDriver:
			device = torch.device("cpu")
			episodes_trained = 0

			def __init__(self, _observation_size: int) -> None:
				self.observation_size = _observation_size
				self.recorded = 0

			def choose_action(self, _observation):
				self.last_observation = _observation
				return 6, torch.tensor(0.0, requires_grad=True)

			def record(self, _log_probability, _reward: float) -> None:
				self.last_record = (_log_probability, _reward)
				self.recorded += 1

			def finish_episode(self):
				return 0.0

		canvas = FakeCanvas()
		root = FakeRoot(canvas)
		fake_tk = types.ModuleType("tkinter")

		def make_canvas(*args, **kwargs):
			canvas.options = (args, kwargs)
			return canvas

		setattr(fake_tk, "Tk", lambda: root)
		setattr(fake_tk, "Canvas", make_canvas)
		times = itertools.count()
		original_tk = sys.modules.get("tkinter")
		sys.modules["tkinter"] = fake_tk
		try:
			with patch("car_ai.OnlineDriver", FakeDriver), patch(
				"time.perf_counter",
				side_effect=lambda: next(times) / 2.0,
			):
				from car import run_gui

				run_gui()
		finally:
			if original_tk is not None:
				sys.modules["tkinter"] = original_tk
			else:
				del sys.modules["tkinter"]

		self.assertEqual(root.window_title, "Procedural Traffic AI")
		self.assertEqual(root.window_geometry, "800x600")
		self.assertGreaterEqual(len(canvas.items), 10)
		texts = [item[2]["text"] for item in canvas.items if item[0] == "text"]
		self.assertTrue(any(text.startswith("EPISODE") for text in texts))
		self.assertTrue(any(text.startswith("DISTANCE") for text in texts))
		self.assertTrue(any(text.startswith("LAST ") for text in texts))
		player_shapes = [
			item
			for frame in (*canvas.frames, canvas.items)
			for item in frame
			if item[0] == "polygon" and item[2].get("fill") == "#e53935"
		]
		self.assertTrue(player_shapes)
		self.assertGreater(
			max(
				sum(
					shape[1][index]
					for index in range(0, len(shape[1]), 2)
				) / (len(shape[1]) / 2)
				for shape in player_shapes
			),
			400.0,
		)


if __name__ == "__main__":
	unittest.main()
