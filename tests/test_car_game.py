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

from car import main
from car_ai import DrivingPolicy, EpisodeRollout, OnlineDriver, SharedPolicyTrainer
from car_simulation import DrivingSimulation, TrafficCar


class CommandLineTests(unittest.TestCase):
	def test_until_stopped_trains_continuously_and_handles_ctrl_c(self) -> None:
		with (
			patch(
				"sys.argv",
				["car.py", "--mode", "headless", "--device", "cpu", "--until-stopped"],
			),
			patch("car_training.train_headless", side_effect=KeyboardInterrupt) as train,
		):
			main()

		self.assertIsNone(train.call_args.kwargs["updates"])


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
		self.assertAlmostEqual(
			observation[simulation.STEERING_STATE_INDEX],
			simulation.applied_steering,
		)
		self.assertAlmostEqual(
			observation[simulation.SPEED_LIMIT_INDEX],
			simulation._safe_speed_limit() / simulation.PLAYER_MAX_SPEED,
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

	def test_sensor_zone_steers_away_from_traffic_and_road_edge(self) -> None:
		simulation = DrivingSimulation(traffic_count=1, seed=8)
		simulation.player.distance = 30.0
		simulation.player.lateral = 0.0
		simulation.traffic[0].distance = 34.0
		simulation.traffic[0].lateral = 2.0

		before = simulation.player.lateral
		simulation.step(action=4, dt=0.1)
		self.assertLess(simulation.player.lateral, before)

		simulation.reset()
		simulation.player.lateral = (
			-simulation.track.track_width / 2.0
			+ simulation.PLAYER_HALF_WIDTH
			+ 0.5
		)
		simulation.traffic[0].distance = simulation.player.distance + 8.0
		simulation.traffic[0].lateral = simulation.player.lateral + 1.0
		before = simulation.player.lateral
		simulation.step(action=4, dt=0.1)
		self.assertGreater(simulation.player.lateral, before)
		self.assertIsNone(simulation.crash_reason)

	def test_steering_changes_direction_smoothly(self) -> None:
		simulation = DrivingSimulation(traffic_count=0, seed=10)

		simulation.step(action=8, dt=0.1)
		lateral_after_right_steer = simulation.player.lateral
		self.assertGreater(simulation.applied_steering, 0.0)

		simulation.step(action=0, dt=0.1)
		self.assertEqual(simulation.applied_steering, 0.0)
		self.assertEqual(simulation.player.lateral, lateral_after_right_steer)

		simulation.step(action=0, dt=0.1)
		self.assertLess(simulation.applied_steering, 0.0)

	def test_traffic_sensor_can_choose_both_steering_directions(self) -> None:
		simulation = DrivingSimulation(traffic_count=1, seed=13)
		car = simulation.traffic[0]
		car.distance = simulation.player.distance + 10.0
		car.lateral = 2.0
		self.assertEqual(simulation._sensor_steering(0.0, 0.1), -1.0)

		car.lateral = -2.0
		self.assertEqual(simulation._sensor_steering(0.0, 0.1), 1.0)

	def test_collision_sensor_selects_the_unoccupied_passing_side(self) -> None:
		simulation = DrivingSimulation(traffic_count=2, seed=18)
		for car in simulation.traffic:
			car.distance = simulation.player.distance + 12.0
		simulation.traffic[0].lateral = 0.0
		simulation.traffic[1].lateral = 3.2

		self.assertEqual(simulation._sensor_steering(0.0, 0.1), -1.0)

	def test_collision_sensor_brakes_when_closing_on_same_lane_traffic(self) -> None:
		simulation = DrivingSimulation(traffic_count=1, seed=19)
		car = simulation.traffic[0]
		car.distance = simulation.player.distance + 8.0
		car.lateral = simulation.player.lateral
		simulation.player.speed = 22.0
		initial_speed = simulation.player.speed

		simulation.step(action=5, dt=0.1)

		self.assertLess(simulation.player.speed, initial_speed)
		self.assertGreaterEqual(simulation.player.speed, simulation.PLAYER_MIN_SPEED)

	def test_reward_penalizes_precrash_collision_risk_and_override(self) -> None:
		safe = DrivingSimulation(traffic_count=0, seed=23)
		at_risk = DrivingSimulation(traffic_count=1, seed=23)
		car = at_risk.traffic[0]
		car.distance = at_risk.player.distance + 8.0
		car.lateral = at_risk.player.lateral

		safe_result = safe.step(action=4, dt=0.1)
		risk_result = at_risk.step(action=4, dt=0.1)

		self.assertFalse(risk_result.done)
		self.assertLess(risk_result.reward, safe_result.reward)

	def test_collision_sensor_predicts_lateral_overlap_at_contact(self) -> None:
		simulation = DrivingSimulation(traffic_count=1, seed=20)
		car = simulation.traffic[0]
		car.distance = simulation.player.distance + 12.0
		car.lateral = 2.8
		simulation.player.speed = 22.0
		simulation.applied_steering = 0.5

		self.assertGreater(simulation._collision_brake(), 0.0)

	def test_swept_collision_detects_midframe_lane_crossing(self) -> None:
		simulation = DrivingSimulation(traffic_count=1, seed=21)
		longitudinal_overlap = simulation._swept_overlap_interval(
			25.0,
			-12.0,
			simulation.COLLISION_LONGITUDINAL_RADIUS,
		)
		lateral_overlap = simulation._swept_overlap_interval(
			4.0,
			-4.9,
			simulation.COLLISION_LATERAL_RADIUS,
		)

		self.assertIsNotNone(longitudinal_overlap)
		self.assertIsNotNone(lateral_overlap)
		self.assertLess(
			max(longitudinal_overlap[0], lateral_overlap[0]),
			min(longitudinal_overlap[1], lateral_overlap[1]),
		)

	def test_oriented_vehicle_collision_includes_exact_edge_touch(self) -> None:
		simulation = DrivingSimulation(traffic_count=0, seed=22)
		contact_distance = simulation.COLLISION_LONGITUDINAL_RADIUS

		self.assertTrue(
			simulation._cars_touch_during_step(
				0.0,
				0.0,
				contact_distance + 1.0,
				contact_distance,
				0.0,
			)
		)
		self.assertFalse(
			simulation._cars_touch_during_step(
				0.0,
				0.0,
				contact_distance + 1.001,
				contact_distance + 0.001,
				0.0,
			)
		)

	def test_speed_reward_penalizes_exceeding_curve_speed_limit(self) -> None:
		within_limit = DrivingSimulation(traffic_count=0, seed=12)
		over_limit = DrivingSimulation(traffic_count=0, seed=12)
		speed_limit = within_limit._safe_speed_limit()
		within_limit.player.speed = speed_limit
		over_limit.player.speed = within_limit.PLAYER_MAX_SPEED

		within_result = within_limit.step(action=4, dt=0.1)
		over_result = over_limit.step(action=4, dt=0.1)

		self.assertLess(over_result.reward, within_result.reward)

	def test_lane_assist_recenters_car_when_policy_is_neutral(self) -> None:
		simulation = DrivingSimulation(traffic_count=0, seed=16)
		simulation.player.lateral = 2.0
		initial_offset = abs(simulation.player.lateral)

		for _ in range(120):
			result = simulation.step(action=4, dt=1.0 / 30.0)
			if result.done:
				self.fail("lane assist should keep the car on the road")

		self.assertLess(abs(simulation.player.lateral), initial_offset)

	def test_speed_assist_brakes_when_above_safe_target(self) -> None:
		simulation = DrivingSimulation(traffic_count=0, seed=17)
		simulation.player.speed = simulation.PLAYER_MAX_SPEED
		initial_speed = simulation.player.speed

		simulation.step(action=5, dt=0.1)

		self.assertLess(simulation.player.speed, initial_speed)
		self.assertGreaterEqual(simulation.player.speed, simulation.PLAYER_MIN_SPEED)


class OnlineDriverTests(unittest.TestCase):
	def test_online_driver_loads_newest_shared_training_checkpoint(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			checkpoint_dir = Path(directory)
			gui_checkpoint = checkpoint_dir / "car_ai_checkpoint_v5.pt"
			shared_checkpoint = checkpoint_dir / "car_ai_shared_checkpoint_v2.pt"
			gui_policy = DrivingPolicy(DrivingSimulation.OBSERVATION_SIZE)
			shared_policy = DrivingPolicy(DrivingSimulation.OBSERVATION_SIZE)
			with torch.no_grad():
				for parameter in gui_policy.parameters():
					parameter.fill_(0.0)
				for parameter in shared_policy.parameters():
					parameter.fill_(0.25)
			torch.save(
				{
					"policy": gui_policy.state_dict(),
					"optimizer": torch.optim.AdamW(gui_policy.parameters()).state_dict(),
					"episodes_trained": 9,
				},
				gui_checkpoint,
			)
			torch.save(
				{
					"version": 2,
					"policy": shared_policy.state_dict(),
					"optimizer": torch.optim.AdamW(shared_policy.parameters()).state_dict(),
					"episodes_trained": 40,
					"updates_completed": 12,
				},
				shared_checkpoint,
			)

			driver = OnlineDriver(
				DrivingSimulation.OBSERVATION_SIZE,
				gui_checkpoint,
			)

			self.assertEqual(driver.loaded_checkpoint, shared_checkpoint)
			self.assertEqual(driver.training_status, "HEADLESS POLICY  |  UPDATE 12")
			for parameter in driver.policy.parameters():
				self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 0.25)))

	def test_online_driver_migrates_legacy_checkpoint_inputs(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			legacy_policy = DrivingPolicy(DrivingSimulation.OBSERVATION_SIZE - 2)
			legacy_weight = legacy_policy.layers[0].weight.detach().clone()
			legacy_checkpoint = Path(directory) / "car_ai_checkpoint_v4.pt"
			legacy_optimizer = torch.optim.AdamW(legacy_policy.parameters())
			torch.save(
				{
					"policy": legacy_policy.state_dict(),
					"optimizer": legacy_optimizer.state_dict(),
					"episodes_trained": 12,
				},
				legacy_checkpoint,
			)

			driver = OnlineDriver(
				DrivingSimulation.OBSERVATION_SIZE,
				Path(directory) / "car_ai_checkpoint_v5.pt",
			)
			migrated_weight = driver.policy.layers[0].weight.detach()

			self.assertEqual(driver.episodes_trained, 12)
			self.assertTrue(torch.equal(migrated_weight[:, :3], legacy_weight[:, :3]))
			self.assertTrue(torch.equal(migrated_weight[:, 5:], legacy_weight[:, 3:]))
			self.assertEqual(torch.count_nonzero(migrated_weight[:, 3:5]).item(), 0)

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
	def test_shared_trainer_migrates_legacy_checkpoint_inputs(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			legacy_policy = DrivingPolicy(DrivingSimulation.OBSERVATION_SIZE - 2)
			legacy_weight = legacy_policy.layers[0].weight.detach().clone()
			legacy_optimizer = torch.optim.AdamW(legacy_policy.parameters())
			legacy_scheduler = torch.optim.lr_scheduler.LambdaLR(
				legacy_optimizer,
				lr_lambda=lambda _update: 1.0,
			)
			legacy_checkpoint = Path(directory) / "car_ai_shared_checkpoint_v1.pt"
			torch.save(
				{
					"version": 1,
					"policy": legacy_policy.state_dict(),
					"optimizer": legacy_optimizer.state_dict(),
					"scheduler": legacy_scheduler.state_dict(),
					"updates_completed": 7,
					"episodes_trained": 21,
					"learning_rate": 3e-4,
					"minimum_learning_rate": 1e-5,
					"learning_rate_decay": 0.995,
					"gamma": 0.99,
				},
				legacy_checkpoint,
			)

			trainer = SharedPolicyTrainer(
				DrivingSimulation.OBSERVATION_SIZE,
				checkpoint_path=Path(directory) / "car_ai_shared_checkpoint_v2.pt",
				device="cpu",
			)
			migrated_weight = trainer.policy.layers[0].weight.detach()

			self.assertEqual(trainer.updates_completed, 7)
			self.assertEqual(trainer.episodes_trained, 21)
			self.assertTrue(torch.equal(migrated_weight[:, :3], legacy_weight[:, :3]))
			self.assertTrue(torch.equal(migrated_weight[:, 5:], legacy_weight[:, 3:]))
			self.assertEqual(torch.count_nonzero(migrated_weight[:, 3:5]).item(), 0)

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
			training_status = "NEW POLICY  |  EPISODE 0"

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

		self.assertEqual(root.window_title, "Apex Driver Lab")
		self.assertEqual(root.window_geometry, "800x600")
		self.assertGreaterEqual(len(canvas.items), 10)
		texts = [item[2]["text"] for item in canvas.items if item[0] == "text"]
		self.assertTrue(any(text == "APEX / DRIVER LAB" for text in texts))
		self.assertTrue(any(text.startswith("EPISODE") for text in texts))
		self.assertTrue(any(text.startswith("DISTANCE") for text in texts))
		self.assertTrue(
			any(text in ("SENSORS CLEAR", "PROXIMITY ALERT") for text in texts)
		)
		self.assertGreater(
			sum(item[0] == "polygon" for item in canvas.items),
			20,
		)
		self.assertTrue(
			any(
				text.startswith(("LAST ", "RUN AVERAGE:"))
				for text in texts
			)
		)
		player_shapes = [
			item
			for frame in (*canvas.frames, canvas.items)
			for item in frame
			if item[0] == "polygon" and item[2].get("fill") == "#e53935"
		]
		self.assertTrue(player_shapes)
		player_body = next(
			item
			for item in canvas.items
			if item[0] == "polygon" and item[2].get("fill") == "#e53935"
		)
		body_points = list(zip(player_body[1][::2], player_body[1][1::2]))
		nose_x = sum(point[0] for point in sorted(body_points, key=lambda point: point[1])[:2]) / 2.0
		body_center_x = sum(point[0] for point in body_points) / len(body_points)
		self.assertGreater(nose_x, body_center_x)
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
