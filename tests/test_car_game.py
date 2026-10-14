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

from car_ai import OnlineDriver
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
			self.assertTrue(checkpoint.is_file())
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

	def test_episode_with_no_actions_does_not_update(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			driver = OnlineDriver(
				DrivingSimulation.OBSERVATION_SIZE,
				Path(directory) / "unused.pt",
			)

			self.assertIsNone(driver.finish_episode())
			self.assertEqual(driver.episodes_trained, 0)


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
