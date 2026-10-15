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
	TRAFFIC_FEATURES_PER_CAR = 4
	ROAD_CLEARANCE_FEATURES = 2
	HORIZONTAL_DISTANCE_SCALE = 40.0
	VERTICAL_DISTANCE_SCALE = 100.0
	LATERAL_DISTANCE_SCALE = 6.0
	RELATIVE_SPEED_SCALE = 20.0
	OBSERVATION_SIZE = (
		3 + ROAD_CLEARANCE_FEATURES
		+ TRAFFIC_FEATURES * TRAFFIC_FEATURES_PER_CAR
	)
	LEFT_CLEARANCE_INDEX = 3
	RIGHT_CLEARANCE_INDEX = 4
	TRAFFIC_FEATURES_START_INDEX = 5
	TRAFFIC_HORIZONTAL_OFFSET = 0
	TRAFFIC_VERTICAL_OFFSET = 1
	TRAFFIC_LANE_OFFSET = 2
	TRAFFIC_RELATIVE_SPEED_OFFSET = 3
	OBSERVATION_V2_SLOT_COUNT = 4
	OBSERVATION_V2_SLOT_FEATURES = 5
	OBSERVATION_V2_SIZE = 4 * 5 + 1 + 1 + 3 + 1
	OBSERVATION_V2_MAX_DISTANCE = 150.0
	OBSERVATION_V2_PROXIMITY_DECAY = 75.0
	OBSERVATION_V2_MAX_TTC = 10.0
	MAX_EPISODE_STEPS = 1800

	def __init__(
		self,
		track: ProceduralTrack | None = None,
		traffic_count: int = 14,
		seed: int = 0,
		obs_v2: bool = False,
		traffic_speed_kmh: float = 20.0,
		ai_min_speed_kmh: float = 40.0,
		randomize_starts: bool = False,
		curriculum: bool = False,
	) -> None:
		if traffic_count < 0:
			raise ValueError("traffic_count cannot be negative")
		if not math.isfinite(traffic_speed_kmh) or traffic_speed_kmh < 0:
			raise ValueError("traffic_speed_kmh must be finite and non-negative")
		if not math.isfinite(ai_min_speed_kmh) or not (
			0 <= ai_min_speed_kmh <= self.PLAYER_MAX_SPEED * 3.6
		):
			raise ValueError("ai_min_speed_kmh must be between 0 and 115.2")
		self.track = track or ProceduralTrack(point_spacing=1.0, chunks_ahead=5)
		self.traffic_density = traffic_count
		self.traffic_count = traffic_count
		self.traffic_speed = traffic_speed_kmh / 3.6
		self.player_min_speed = ai_min_speed_kmh / 3.6
		self.TRAFFIC_SPEED = self.traffic_speed
		self.PLAYER_MIN_SPEED = self.player_min_speed
		self.obs_v2 = obs_v2
		self.randomize_starts = randomize_starts
		self.curriculum = curriculum
		self.mean_training_distance = 0.0
		self._rng = random.Random(seed)
		self._next_car_id = 0
		self.player = Vehicle(distance=0.0, lateral=0.0, speed=self.PLAYER_INITIAL_SPEED)
		self.traffic: List[TrafficCar] = []
		self.steps = 0
		self.crash_reason: str | None = None
		self.reset()

	def _make_traffic_car(self, distance: float, lane: float) -> TrafficCar:
		car = TrafficCar(
			distance=distance,
			lateral=lane,
			speed=self.traffic_speed,
			car_id=self._next_car_id,
		)
		self._next_car_id += 1
		return car

	def reset(self) -> Observation:
		"""Reset the AI car and place a fresh stream of scripted traffic."""
		if self.curriculum:
			stage_density = min(
				self.traffic_density,
				1 + int(max(0.0, self.mean_training_distance) // 100.0),
			)
			self.traffic_count = stage_density
		else:
			self.traffic_count = self.traffic_density

		initial_speed = max(self.player_min_speed, self.PLAYER_INITIAL_SPEED)
		initial_lateral = 0.0
		if self.randomize_starts:
			initial_speed = self._rng.uniform(
				self.player_min_speed, self.PLAYER_MAX_SPEED
			)
			max_lateral = max(
				0.0,
				self.track.track_width / 2.0 - self.PLAYER_HALF_WIDTH - 0.25,
			)
			initial_lateral = self._rng.uniform(-max_lateral, max_lateral)
		self.player = Vehicle(
			distance=0.0,
			lateral=initial_lateral,
			speed=initial_speed,
		)
		self.steps = 0
		self.crash_reason = None
		self.traffic = []
		lanes = (-3.2, 0.0, 3.2)
		near_count = 0
		if self.curriculum:
			near_count = self._rng.randint(0, self.traffic_count)
		near_distances = sorted(
			self._rng.uniform(15.0, 100.0) for _ in range(near_count)
		)
		random_first_spawn = self._rng.uniform(15.0, 80.0)
		for index in range(self.traffic_count):
			if index < near_count:
				distance = near_distances[index]
			elif self.randomize_starts:
				distance = 100.0 + (index - near_count) * 20.0 + self._rng.uniform(0, 20)
				if not self.curriculum:
					distance = random_first_spawn + index * 20.0 + self._rng.uniform(0, 20)
			else:
				distance = 24.0 + index * 20.0
			lane = self._rng.choice(lanes) if self.randomize_starts else lanes[index % len(lanes)]
			self.traffic.append(self._make_traffic_car(distance, lane))
		return self.observe()

	def observe(self) -> Observation:
		if self.obs_v2:
			return self.observe_v2()

		"""Return normalized inputs: road data then (x, y, lane, speed) per car.

			Traffic x is world-horizontal meters, positive to the right; y is
			track-forward meters, positive for a car ahead. Lane is the signed
			track-relative lateral separation. Distances are scaled by their named
			scales and clamped for stable policy-network inputs.
		"""
		road_half_width = self.track.track_width / 2.0
		speed_limit = self.PLAYER_MAX_SPEED
		player_world_x = (
			self.track.center_x_at(self.player.distance) + self.player.lateral
		)
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
			key=lambda car: (
				abs(car.distance - self.player.distance),
				abs(car.lateral - self.player.lateral),
			),
		)[: self.TRAFFIC_FEATURES]
		for car in nearby:
			car_world_x = self.track.center_x_at(car.distance) + car.lateral
			horizontal_distance = car_world_x - player_world_x
			vertical_distance = car.distance - self.player.distance
			features.extend(
				(
					max(
						-1.0,
						min(1.0, horizontal_distance / self.HORIZONTAL_DISTANCE_SCALE),
					),
					max(
						-1.0,
						min(1.0, vertical_distance / self.VERTICAL_DISTANCE_SCALE),
					),
					max(
						-1.0,
						min(
							1.0,
							(car.lateral - self.player.lateral)
							/ self.LATERAL_DISTANCE_SCALE,
						),
					),
					max(
						-1.0,
						min(
							1.0,
							(car.speed - self.player.speed) / self.RELATIVE_SPEED_SCALE,
						),
					),
				)
			)
		for _ in range(self.TRAFFIC_FEATURES - len(nearby)):
			features.extend((0.0, 1.0, 0.0, 0.0))

		return tuple(features)

	def _road_heading(self, distance: float) -> float:
		"""Estimate centerline heading relative to the track-forward axis."""
		before_distance = max(0.0, distance - 1.0)
		before = self.track.center_x_at(before_distance)
		after = self.track.center_x_at(distance + 1.0)
		return math.atan2(after - before, distance + 1.0 - before_distance)

	def _road_curvature(self, distance: float) -> float:
		"""Estimate signed centerline curvature in inverse meters."""
		if distance < 1.0:
			before = self.track.center_x_at(distance)
			center = self.track.center_x_at(distance + 1.0)
			after = self.track.center_x_at(distance + 2.0)
			slope = center - before
			second_derivative = after - 2.0 * center + before
			return second_derivative / (1.0 + slope * slope) ** 1.5
		before = self.track.center_x_at(distance - 1.0)
		center = self.track.center_x_at(distance)
		after = self.track.center_x_at(distance + 1.0)
		slope = (after - before) / 2.0
		second_derivative = after - 2.0 * center + before
		return second_derivative / (1.0 + slope * slope) ** 1.5

	def _observation_v2_slot(
		self,
		car: TrafficCar | None,
	) -> tuple[float, float, float, float, float]:
		if car is None:
			return (1.0, 0.0, 0.0, 0.0, 0.0)

		longitudinal = car.distance - self.player.distance
		lateral = car.lateral - self.player.lateral
		relative_speed = car.speed - self.player.speed
		proximity = math.exp(
			-abs(longitudinal) / self.OBSERVATION_V2_PROXIMITY_DECAY
		)
		normalized_distance = max(
			-1.0,
			min(1.0, longitudinal / self.OBSERVATION_V2_MAX_DISTANCE),
		) * proximity
		normalized_lateral = max(
			-1.0,
			min(1.0, lateral / (self.track.track_width / 2.0)),
		) * proximity
		normalized_relative_speed = max(
			-1.0,
			min(1.0, relative_speed / self.PLAYER_MAX_SPEED),
		) * proximity

		closing_speed = (
			self.player.speed - car.speed
			if longitudinal >= 0
			else car.speed - self.player.speed
		)
		time_to_collision = (
			abs(longitudinal) / closing_speed
			if closing_speed > 1e-6
			else math.inf
		)
		normalized_ttc = (
			max(
				0.0,
				1.0 - min(
					1.0,
					time_to_collision / self.OBSERVATION_V2_MAX_TTC,
				),
			)
			* proximity
			if math.isfinite(time_to_collision)
			else 0.0
		)
		return (
			normalized_distance,
			normalized_lateral,
			normalized_relative_speed,
			normalized_ttc,
			1.0,
		)

	def observe_v2(self) -> Observation:
		"""Return ego-relative fixed traffic slots followed by road/speed inputs."""
		slots: list[TrafficCar | None] = [None, None, None, None]
		candidates: list[list[TrafficCar]] = [[], [], [], []]
		for car in self.traffic:
			longitudinal = car.distance - self.player.distance
			lateral = car.lateral - self.player.lateral
			if longitudinal < 0:
				candidates[3].append(car)
			elif abs(lateral) < 1.6:
				candidates[0].append(car)
			elif lateral < 0:
				candidates[1].append(car)
			else:
				candidates[2].append(car)

		for index in range(3):
			if candidates[index]:
				slots[index] = min(
					candidates[index],
					key=lambda car: car.distance - self.player.distance,
				)
		if candidates[3]:
			slots[3] = max(
				candidates[3],
				key=lambda car: car.distance - self.player.distance,
			)

		features: list[float] = []
		for car in slots:
			features.extend(self._observation_v2_slot(car))

		road_half_width = self.track.track_width / 2.0
		heading = (
			self._road_heading(self.player.distance + 10.0)
			- self._road_heading(self.player.distance)
		)
		features.extend(
			(
				max(-1.0, min(1.0, self.player.lateral / road_half_width)),
				max(-1.0, min(1.0, heading / (math.pi / 4.0))),
			)
		)
		for lookahead in (10.0, 20.0, 40.0):
			curvature = self._road_curvature(self.player.distance + lookahead)
			features.append(max(-1.0, min(1.0, curvature / 0.03)))
		speed_range = self.PLAYER_MAX_SPEED - self.player_min_speed
		normalized_speed = (
			2.0 * (self.player.speed - self.player_min_speed) / speed_range - 1.0
			if speed_range > 0
			else 0.0
		)
		features.append(max(-1.0, min(1.0, normalized_speed)))
		return tuple(features)

	def set_curriculum_distance(self, mean_distance: float) -> None:
		"""Provide the recent mean completion distance for the next reset stage."""
		if not math.isfinite(mean_distance) or mean_distance < 0:
			raise ValueError("mean_distance must be finite and non-negative")
		self.mean_training_distance = mean_distance

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
			self.player_min_speed,
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
