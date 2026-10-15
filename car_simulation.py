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
	PLAYER_HALF_LENGTH = 2.0
	LATERAL_SPEED = 8.0
	STEERING_RATE = 1.5
	LANE_CENTERING_GAIN = 0.08
	STEERING_DAMPING = 0.55
	AI_STEERING_INFLUENCE = 0.2
	AI_THROTTLE_INFLUENCE = 0.15
	PLAYER_ACCELERATION = 4.0
	PLAYER_BRAKING = 8.0
	SPEED_CONTROL_GAIN = 0.5
	PLAYER_CRUISE_SPEED = 22.0
	CURVE_SPEED_REDUCTION = 6.0
	OVERSPEED_PENALTY = 0.02
	COLLISION_RISK_PENALTY = 0.3
	EDGE_RISK_PENALTY = 0.15
	SAFETY_OVERRIDE_PENALTY = 0.1
	SENSOR_RADIUS = 18.0
	SENSOR_WIDTH = 6.0
	SENSOR_MARGIN = 1.5
	COLLISION_LONGITUDINAL_RADIUS = 2.0 * PLAYER_HALF_LENGTH
	COLLISION_LATERAL_RADIUS = 2.0 * PLAYER_HALF_WIDTH
	REACTION_TIME = 1.0
	COLLISION_LOOKAHEAD = 2.0
	ROAD_BOUNDARY_EPSILON = 0.05
	STEERING_VALUES = (-1.0, 0.0, 1.0)
	THROTTLE_VALUES = (-1.0, 0.0, 1.0)
	TRAFFIC_FEATURES = 4
	TRAFFIC_FEATURES_PER_CAR = 4
	ROAD_CLEARANCE_FEATURES = 2
	CONTROL_FEATURES = 2
	HORIZONTAL_DISTANCE_SCALE = 40.0
	VERTICAL_DISTANCE_SCALE = 100.0
	LATERAL_DISTANCE_SCALE = 6.0
	RELATIVE_SPEED_SCALE = 20.0
	OBSERVATION_SIZE = (
		3 + CONTROL_FEATURES + ROAD_CLEARANCE_FEATURES
		+ TRAFFIC_FEATURES * TRAFFIC_FEATURES_PER_CAR
	)
	STEERING_STATE_INDEX = 3
	SPEED_LIMIT_INDEX = 4
	LEFT_CLEARANCE_INDEX = 5
	RIGHT_CLEARANCE_INDEX = 6
	TRAFFIC_FEATURES_START_INDEX = 7
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
	MAX_EPISODE_STEPS: int = 1800

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
		self.applied_steering = 0.0
		self.player_lateral_velocity = 0.0
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
		self.applied_steering = 0.0
		self.player_lateral_velocity = 0.0
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
		speed_scale = self.PLAYER_MAX_SPEED
		safe_speed_limit = self._safe_speed_limit()
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
			max(0.0, min(1.0, self.player.speed / speed_scale)),
			max(-1.0, min(1.0, curve_ahead)),
			self.applied_steering,
			max(0.0, min(1.0, safe_speed_limit / speed_scale)),
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

	def _safe_speed_limit(self) -> float:
		"""Return a soft speed target that decreases for tighter upcoming curves."""
		start = self.track.center_x_at(self.player.distance)
		middle = self.track.center_x_at(self.player.distance + 8.0)
		end = self.track.center_x_at(self.player.distance + 16.0)
		curve_load = abs(end - 2.0 * middle + start)
		curve_reduction = min(
			self.PLAYER_CRUISE_SPEED - self.PLAYER_MIN_SPEED,
			curve_load * self.CURVE_SPEED_REDUCTION,
		)
		return max(
			self.PLAYER_MIN_SPEED,
			self.PLAYER_CRUISE_SPEED - curve_reduction,
		)

	def _update_traffic(self, dt: float) -> None:
		for car in self.traffic:
			car.distance += car.speed * dt
		self.traffic = [
			car for car in self.traffic
			if car.distance >= self.player.distance - 55.0
		]
		lanes = (-3.2, 0.0, 3.2)
		while len(self.traffic) < self.traffic_count:
			furthest_distance = self.player.distance + 80.0
			for car in self.traffic:
				furthest_distance = max(furthest_distance, car.distance)
			self.traffic.append(
				self._make_traffic_car(
					furthest_distance + self._rng.uniform(20.0, 38.0),
					lanes[self._next_car_id % len(lanes)],
				)
			)

	def _sensor_steering(self, requested_steering: float, dt: float) -> float:
		"""Avoid nearby traffic while keeping the car inside the road boundaries."""
		road_half_width = self.track.track_width / 2.0
		if abs(self.player.lateral) + self.PLAYER_HALF_WIDTH >= road_half_width:
			return requested_steering

		nearby_cars = []
		for car in self.traffic:
			relative_distance = car.distance - self.player.distance
			relative_lateral = car.lateral - self.player.lateral
			if (
				0.0 < relative_distance <= self.SENSOR_RADIUS
				and abs(relative_lateral) <= self.SENSOR_WIDTH
			):
				nearby_cars.append(car)
		if nearby_cars:
			max_lateral = road_half_width - self.PLAYER_HALF_WIDTH
			candidate_scores = {}
			for direction in (-1.0, 1.0):
				candidate_lateral = max(
					-max_lateral,
					min(max_lateral, self.player.lateral + direction * 2.8),
				)
				edge_clearance = min(
					candidate_lateral + max_lateral,
					max_lateral - candidate_lateral,
				)
				traffic_clearance = min(
					abs(car.lateral - candidate_lateral)
					- self.COLLISION_LATERAL_RADIUS
					for car in nearby_cars
				)
				candidate_scores[direction] = min(edge_clearance, traffic_clearance)
			requested_steering = max(
				candidate_scores,
				key=lambda direction: candidate_scores[direction],
			)

		max_lateral = (
			road_half_width - self.PLAYER_HALF_WIDTH - self.ROAD_BOUNDARY_EPSILON
		)
		left_edge_margin = road_half_width + self.player.lateral - self.PLAYER_HALF_WIDTH
		right_edge_margin = road_half_width - self.player.lateral - self.PLAYER_HALF_WIDTH
		if (
			left_edge_margin <= self.SENSOR_MARGIN
			and requested_steering <= 0.0
		):
			requested_steering = 1.0
		if (
			right_edge_margin <= self.SENSOR_MARGIN
			and requested_steering >= 0.0
		):
			requested_steering = -1.0

		next_lateral = (
			self.player.lateral + requested_steering * self.LATERAL_SPEED * dt
		)
		if next_lateral < -max_lateral:
			return 1.0
		if next_lateral > max_lateral:
			return -1.0
		return requested_steering

	def _collision_brake(self) -> float:
		"""Estimate braking urgency from time-to-contact and predicted lane position."""
		urgency = 0.0
		for car in self.traffic:
			gap = car.distance - self.player.distance
			if not 0.0 < gap <= self.SENSOR_RADIUS:
				continue
			closing_speed = max(0.0, self.player.speed - car.speed)
			if closing_speed <= 0.0:
				time_to_contact = self.COLLISION_LOOKAHEAD
			else:
				time_to_contact = max(
					0.0,
					(gap - self.COLLISION_LONGITUDINAL_RADIUS) / closing_speed,
				)
			if time_to_contact > self.COLLISION_LOOKAHEAD:
				continue
			predicted_lateral = (
				self.player.lateral
				+ self.applied_steering * self.LATERAL_SPEED * time_to_contact
			)
			lateral_gap = abs(car.lateral - predicted_lateral)
			if lateral_gap >= self.COLLISION_LATERAL_RADIUS:
				continue
			required_time = max(self.REACTION_TIME, self.COLLISION_LOOKAHEAD)
			car_urgency = max(
				0.0,
				min(1.0, (required_time - time_to_contact) / required_time),
			)
			urgency = max(urgency, car_urgency)
		return urgency

	@staticmethod
	def _swept_overlap_interval(
		start: float,
		end: float,
		radius: float,
	) -> Tuple[float, float] | None:
		"""Return the timestep interval where a linear coordinate is within radius."""
		delta = end - start
		if delta == 0.0:
			return (0.0, 1.0) if abs(start) <= radius else None
		first = (-radius - start) / delta
		last = (radius - start) / delta
		enter = max(0.0, min(first, last))
		exit = min(1.0, max(first, last))
		return (enter, exit) if enter <= exit else None

	def _cars_touch_during_step(
		self,
		start_lateral_gap: float,
		end_lateral_gap: float,
		start_distance_gap: float,
		end_distance_gap: float,
		heading: float,
	) -> bool:
		"""Sweep the two oriented rectangular car footprints and include edge contact."""
		player_forward = (math.sin(heading), math.cos(heading))
		player_side = (math.cos(heading), -math.sin(heading))
		traffic_forward = (0.0, 1.0)
		traffic_side = (1.0, 0.0)
		axes = (player_forward, player_side, traffic_forward, traffic_side)
		start_position = (start_lateral_gap, start_distance_gap)
		end_position = (end_lateral_gap, end_distance_gap)
		contact_start = 0.0
		contact_end = 1.0

		for axis_x, axis_y in axes:
			player_radius = (
				self.PLAYER_HALF_LENGTH
				* abs(player_forward[0] * axis_x + player_forward[1] * axis_y)
				+ self.PLAYER_HALF_WIDTH
				* abs(player_side[0] * axis_x + player_side[1] * axis_y)
			)
			traffic_radius = (
				self.PLAYER_HALF_LENGTH
				* abs(traffic_forward[0] * axis_x + traffic_forward[1] * axis_y)
				+ self.PLAYER_HALF_WIDTH
				* abs(traffic_side[0] * axis_x + traffic_side[1] * axis_y)
			)
			start_projection = (
				start_position[0] * axis_x + start_position[1] * axis_y
			)
			end_projection = (
				end_position[0] * axis_x + end_position[1] * axis_y
			)
			interval = self._swept_overlap_interval(
				start_projection,
				end_projection,
				player_radius + traffic_radius,
			)
			if interval is None:
				return False
			contact_start = max(contact_start, interval[0])
			contact_end = min(contact_end, interval[1])
			if contact_start > contact_end:
				return False
		return True

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
		steering_target = self._sensor_steering(steering, dt)
		collision_urgency = self._collision_brake()
		lane_assist = max(
			-0.35,
			min(
				0.35,
				-self.player.lateral * self.LANE_CENTERING_GAIN
				- self.applied_steering * self.STEERING_DAMPING,
			),
		)
		if steering_target != steering:
			steering_target = 0.85 * steering_target + 0.15 * lane_assist
		else:
			steering_target = (
				self.AI_STEERING_INFLUENCE * steering_target
				+ (1.0 - self.AI_STEERING_INFLUENCE) * lane_assist
			)
		safe_speed_limit = self._safe_speed_limit()
		speed_error = safe_speed_limit - self.player.speed
		speed_assist = max(-1.0, min(1.0, speed_error * self.SPEED_CONTROL_GAIN))
		throttle_target = (
			self.AI_THROTTLE_INFLUENCE * throttle
			+ (1.0 - self.AI_THROTTLE_INFLUENCE) * speed_assist
		)
		throttle_target = min(throttle_target, -collision_urgency)
		if throttle < 0.0:
			throttle_target = min(throttle_target, -0.5)
		max_steering_change = self.STEERING_RATE * dt
		steering_change = max(
			-max_steering_change,
			min(max_steering_change, steering_target - self.applied_steering),
		)
		self.applied_steering += steering_change
		if abs(self.player.lateral) + self.PLAYER_HALF_WIDTH < self.track.track_width / 2.0:
			max_lateral = (
				self.track.track_width / 2.0
				- self.PLAYER_HALF_WIDTH
				- self.ROAD_BOUNDARY_EPSILON
			)
			next_lateral = (
				self.player.lateral
				+ self.applied_steering * self.LATERAL_SPEED * dt
			)
			if next_lateral > max_lateral:
				self.applied_steering = (
					max_lateral - self.player.lateral
				) / (self.LATERAL_SPEED * dt)
			elif next_lateral < -max_lateral:
				self.applied_steering = (
					-max_lateral - self.player.lateral
				) / (self.LATERAL_SPEED * dt)
		previous_distance = self.player.distance
		previous_lateral = self.player.lateral
		self.player.lateral += self.applied_steering * self.LATERAL_SPEED * dt
		self.player_lateral_velocity = (
			self.player.lateral - previous_lateral
		) / dt
		acceleration = (
			self.PLAYER_ACCELERATION
			if throttle_target >= 0.0
			else self.PLAYER_BRAKING
		)
		self.player.speed = max(
			self.player_min_speed,
			min(
				self.PLAYER_MAX_SPEED,
				self.player.speed + throttle_target * acceleration * dt,
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
				start_gap = car.distance - car.speed * dt - previous_distance
				end_gap = car.distance - self.player.distance
				collided_during_step = self._cars_touch_during_step(
					car.lateral - previous_lateral,
					car.lateral - self.player.lateral,
					start_gap,
					end_gap,
					math.atan2(self.player_lateral_velocity, self.player.speed),
				)
				if collided_during_step:
					reason = "traffic collision"
					break
		if reason is None and self.steps >= self.MAX_EPISODE_STEPS:
			reason = "episode limit"

		reward = max(0.0, self.player.distance - previous_distance) * 0.08
		reward += min(self.player.speed, safe_speed_limit) * 0.002
		reward -= max(0.0, self.player.speed - safe_speed_limit) * self.OVERSPEED_PENALTY
		reward -= abs(self.player.lateral) / (self.track.track_width / 2.0) * 0.01
		road_clearance = (
			self.track.track_width / 2.0
			- self.PLAYER_HALF_WIDTH
			- abs(self.player.lateral)
		)
		edge_risk = max(
			0.0,
			min(1.0, (self.SENSOR_MARGIN - road_clearance) / self.SENSOR_MARGIN),
		)
		reward -= collision_urgency * self.COLLISION_RISK_PENALTY
		reward -= edge_risk * self.EDGE_RISK_PENALTY
		reward -= (
			abs(steering - steering_target) * self.SAFETY_OVERRIDE_PENALTY
		)
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
