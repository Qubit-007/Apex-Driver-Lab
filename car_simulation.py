"""Headless driving simulation for the procedural-track game."""

from dataclasses import dataclass
import math
import random
from typing import List, Tuple

from car import ProceduralTrack

Observation = Tuple[float, ...]


@dataclass
class Vehicle:
	distance: float
	lateral: float
	speed: float


@dataclass
class TrafficCar(Vehicle):
	car_id: int


@dataclass(frozen=True)
class StepResult:
	observation: Observation
	reward: float
	done: bool
	crash_reason: str | None


class DrivingSimulation:
	"""Track-relative driving dynamics, scripted traffic, and terminal events."""

	TRAFFIC_SPEED = 20.0 / 3.6
	PLAYER_MIN_SPEED = 40.0 / 3.6
	PLAYER_INITIAL_SPEED = 18.0
	PLAYER_MAX_SPEED = 32.0
	PLAYER_HALF_WIDTH = 1.1
	STEERING_VALUES = (-1.0, 0.0, 1.0)
	THROTTLE_VALUES = (-1.0, 0.0, 1.0)
	TRAFFIC_FEATURES = 4
	ROAD_CLEARANCE_FEATURES = 2
	OBSERVATION_SIZE = 3 + ROAD_CLEARANCE_FEATURES + TRAFFIC_FEATURES * 3
	LEFT_CLEARANCE_INDEX = 3
	RIGHT_CLEARANCE_INDEX = 4
	TRAFFIC_FEATURES_START_INDEX = 5
	MAX_EPISODE_STEPS = 1800

	def __init__(
		self,
		track: ProceduralTrack | None = None,
		traffic_count: int = 14,
		seed: int = 0,
	) -> None:
		if traffic_count < 0:
			raise ValueError("traffic_count cannot be negative")
		self.track = track or ProceduralTrack(point_spacing=1.0, chunks_ahead=5)
		self.traffic_count = traffic_count
		self._rng = random.Random(seed)
		self._next_car_id = 0
		self.player = Vehicle(distance=0.0, lateral=0.0, speed=18.0)
		self.traffic: List[TrafficCar] = []
		self.steps = 0
		self.crash_reason: str | None = None
		self.reset()

	def _make_traffic_car(self, distance: float, lane: float) -> TrafficCar:
		car = TrafficCar(
			distance=distance,
			lateral=lane,
			speed=self.TRAFFIC_SPEED,
			car_id=self._next_car_id,
		)
		self._next_car_id += 1
		return car

	def reset(self) -> Observation:
		"""Reset the AI car and place a fresh stream of scripted traffic."""
		self.player = Vehicle(
			distance=0.0,
			lateral=0.0,
			speed=max(self.PLAYER_MIN_SPEED, self.PLAYER_INITIAL_SPEED),
		)
		self.steps = 0
		self.crash_reason = None
		self.traffic = []
		lanes = (-3.2, 0.0, 3.2)
		for index in range(self.traffic_count):
			distance = 24.0 + index * 20.0
			lane = lanes[index % len(lanes)]
			self.traffic.append(self._make_traffic_car(distance, lane))
		return self.observe()

	def observe(self) -> Observation:
		"""Describe road edges and nearest traffic using normalized relative values."""
		road_half_width = self.track.track_width / 2.0
		speed_limit = self.PLAYER_MAX_SPEED
		curve_ahead = (
			self.track.center_x_at(self.player.distance + 16.0)
			- self.track.center_x_at(self.player.distance)
		) / road_half_width
		left_clearance = (
			road_half_width + self.player.lateral - self.PLAYER_HALF_WIDTH
		) / self.track.track_width
		right_clearance = (
			road_half_width - self.player.lateral - self.PLAYER_HALF_WIDTH
		) / self.track.track_width
		features = [
			max(-1.0, min(1.0, self.player.lateral / road_half_width)),
			max(0.0, min(1.0, self.player.speed / speed_limit)),
			max(-1.0, min(1.0, curve_ahead)),
			max(0.0, min(1.0, left_clearance)),
			max(0.0, min(1.0, right_clearance)),
		]

		nearby = sorted(
			self.traffic,
			key=lambda car: abs(car.distance - self.player.distance),
		)[: self.TRAFFIC_FEATURES]
		for car in nearby:
			features.extend(
				(
					max(-1.0, min(1.0, (car.distance - self.player.distance) / 100.0)),
					max(-1.0, min(1.0, (car.lateral - self.player.lateral) / road_half_width)),
					max(-1.0, min(1.0, (car.speed - self.player.speed) / 20.0)),
				)
			)
		for _ in range(self.TRAFFIC_FEATURES - len(nearby)):
			features.extend((1.0, 0.0, 0.0))

		return tuple(features)

	def _update_traffic(self, dt: float) -> None:
		for car in self.traffic:
			car.distance += car.speed * dt
		self.traffic = [
			car for car in self.traffic
			if car.distance >= self.player.distance - 55.0
		]
		lanes = (-3.2, 0.0, 3.2)
		while len(self.traffic) < self.traffic_count:
			furthest_distance = max(
				(self.player.distance + 80.0, *(car.distance for car in self.traffic))
			)
			self.traffic.append(
				self._make_traffic_car(
					furthest_distance + self._rng.uniform(20.0, 38.0),
					lanes[self._next_car_id % len(lanes)],
				)
			)

	def step(self, action: int, dt: float) -> StepResult:
		"""Advance one frame; actions encode 3 steering and 3 throttle choices."""
		action_count = len(self.STEERING_VALUES) * len(self.THROTTLE_VALUES)
		if not isinstance(action, int) or not 0 <= action < action_count:
			raise ValueError("action must be an integer from 0 through 8")
		if not math.isfinite(dt) or dt <= 0:
			raise ValueError("dt must be a finite, positive duration")
		if self.crash_reason is not None:
			raise RuntimeError("cannot step a finished episode; call reset() first")

		steering = self.STEERING_VALUES[action // len(self.THROTTLE_VALUES)]
		throttle = self.THROTTLE_VALUES[action % len(self.THROTTLE_VALUES)]
		previous_distance = self.player.distance
		self.player.lateral += steering * 8.0 * dt
		self.player.speed = max(
			self.PLAYER_MIN_SPEED,
			min(
				self.PLAYER_MAX_SPEED,
				self.player.speed + throttle * (10.0 if throttle > 0 else 16.0) * dt,
			),
		)
		self.player.distance += self.player.speed * dt
		self._update_traffic(dt)
		self.steps += 1

		reason = None
		if (
			abs(self.player.lateral) + self.PLAYER_HALF_WIDTH
			>= self.track.track_width / 2.0
		):
			reason = "off the road"
		else:
			for car in self.traffic:
				if (
					abs(car.distance - self.player.distance) < 4.0
					and abs(car.lateral - self.player.lateral) < 2.2
				):
					reason = "traffic collision"
					break
		if reason is None and self.steps >= self.MAX_EPISODE_STEPS:
			reason = "episode limit"

		reward = max(0.0, self.player.distance - previous_distance) * 0.08
		reward += self.player.speed * 0.002
		reward -= abs(self.player.lateral) / (self.track.track_width / 2.0) * 0.01
		if reason == "off the road":
			reward -= 25.0
		elif reason == "traffic collision":
			reward -= 40.0
		elif reason == "episode limit":
			reward += 15.0
		if reason is not None:
			self.crash_reason = reason

		return StepResult(
			observation=self.observe(),
			reward=reward,
			done=reason is not None,
			crash_reason=reason,
		)
