"""On-demand procedural track geometry and an online-learning driving game."""

from dataclasses import dataclass
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class TrackChunk:
	"""Sampled centerline points for one section of track."""

	index: int
	start_s: float
	end_s: float
	points: torch.Tensor  # Shape: (number_of_samples, 2), columns are x and y.
	width: torch.Tensor  # Track width at each sample.


class ProceduralTrack:
	"""Keep the car's current track chunk and a configurable distance ahead.

	Call ``update(car_s)`` as the car moves forward. Generated sections are
	deterministic, so distant chunks can be discarded and recreated later.
	"""

	def __init__(
		self,
		chunk_length: float = 100.0,
		point_spacing: float = 2.0,
		chunks_ahead: int = 4,
		track_width: float = 12.0,
		seed: int = 0,
	) -> None:
		if chunk_length <= 0 or point_spacing <= 0 or track_width <= 0:
			raise ValueError("chunk_length, point_spacing, and track_width must be positive")
		if chunks_ahead < 0:
			raise ValueError("chunks_ahead cannot be negative")

		self.chunk_length = float(chunk_length)
		self.point_spacing = float(point_spacing)
		self.chunks_ahead = int(chunks_ahead)
		self.track_width = float(track_width)
		self._chunks: Dict[int, TrackChunk] = {}

		# Seeded wave parameters ensure that neighboring chunks connect exactly.
		generator = torch.Generator(device="cpu").manual_seed(seed)
		self._amplitudes = torch.tensor([18.0, 9.0, 4.0]) * torch.rand(3, generator=generator)
		self._wavelengths = torch.tensor([260.0, 125.0, 58.0])
		self._phases = torch.rand(3, generator=generator) * (2.0 * math.pi)
		self.update(0.0)

	@property
	def active_chunks(self) -> Tuple[TrackChunk, ...]:
		"""Loaded chunks, ordered from the current section into the future."""
		return tuple(self._chunks[index] for index in sorted(self._chunks))

	def _center_x(self, distance: torch.Tensor) -> torch.Tensor:
		angles = (2.0 * math.pi / self._wavelengths) * distance.unsqueeze(-1)
		angles = angles + self._phases
		return (torch.sin(angles) * self._amplitudes).sum(dim=-1)

	def center_x_at(self, distance: float) -> float:
		"""Return the track centerline's x coordinate at a distance along it."""
		if not math.isfinite(distance) or distance < 0:
			raise ValueError("distance must be a finite, non-negative distance")
		position = torch.tensor(distance, dtype=torch.float32)
		return float(self._center_x(position).item())

	def _generate_chunk(self, index: int) -> TrackChunk:
		start = index * self.chunk_length
		end = start + self.chunk_length
		sample_count = max(1, math.ceil(self.chunk_length / self.point_spacing))
		distance = torch.linspace(start, end, sample_count + 1, dtype=torch.float32)
		points = torch.stack((self._center_x(distance), distance), dim=-1)
		width = torch.full_like(distance, self.track_width)
		return TrackChunk(index, start, end, points, width)

	def update(self, car_s: float) -> Tuple[TrackChunk, ...]:
		"""Stream in the current chunk plus ``chunks_ahead`` future chunks."""
		if not math.isfinite(car_s) or car_s < 0:
			raise ValueError("car_s must be a finite, non-negative distance")

		current = int(car_s // self.chunk_length)
		wanted = set(range(current, current + self.chunks_ahead + 1))
		for index in self._chunks.keys() - wanted:
			del self._chunks[index]
		for index in sorted(wanted - self._chunks.keys()):
			self._chunks[index] = self._generate_chunk(index)
		return self.active_chunks

	def get_chunk(self, index: int) -> TrackChunk:
		"""Get a loaded chunk; call ``update`` to load chunks for a car position."""
		try:
			return self._chunks[index]
		except KeyError as exc:
			raise KeyError(f"chunk {index} is not loaded; call update(car_s) first") from exc


def run_gui(device: str | None = None) -> None:
	"""Watch a PyTorch driver learn to navigate procedural traffic."""
	from collections import deque
	import time
	import tkinter as tk

	from car_ai import OnlineDriver
	from car_simulation import DrivingSimulation

	root = tk.Tk()
	root.title("Apex Driver Lab")
	root.geometry("800x600")
	root.minsize(640, 480)
	root.configure(bg="#17231e")

	canvas = tk.Canvas(root, width=800, height=600, bg="#20352c", highlightthickness=0)
	canvas.pack(fill="both", expand=True)

	track = ProceduralTrack(point_spacing=1.0, chunks_ahead=5)
	simulation = DrivingSimulation(track=track, traffic_count=14, seed=17)
	if device is None:
		driver = OnlineDriver(simulation.OBSERVATION_SIZE)
	else:
		driver = OnlineDriver(simulation.OBSERVATION_SIZE, device=device)
	episode = 1
	recent_runs: deque[tuple[float, int]] = deque(maxlen=20)
	crash_message: str | None = None
	restart_at = 0.0
	last_frame = time.perf_counter()

	def draw_car(
		distance: float,
		lateral: float,
		y: float,
		color: str,
		scale: float,
		canvas_center: float,
		road_center_x: float,
		heading: float = 0.0,
	) -> None:
		center_x = canvas_center + (
			track.center_x_at(distance) + lateral - road_center_x
		) * scale
		half_width = 1.1 * scale
		length = 2.0 * scale
		cos_heading = math.cos(heading)
		sin_heading = math.sin(heading)

		def projected_points(points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
			projected = []
			for local_x, local_y in points:
				projected.extend((
					center_x + local_x * cos_heading - local_y * sin_heading,
					y + local_x * sin_heading + local_y * cos_heading,
				))
			return tuple(projected)

		def draw_polygon(
			points: tuple[tuple[float, float], ...],
			fill: str,
			outline: str = "",
			width: int = 1,
		) -> None:
			canvas.create_polygon(
				*projected_points(points), fill=fill, outline=outline, width=width,
			)

		draw_polygon((
			(-half_width * 0.72, length * 0.78),
			(-half_width * 0.9, length * 0.12),
			(0.0, -length * 0.12),
			(half_width * 0.9, length * 0.12),
			(half_width * 0.72, length * 0.78),
		), "#111714")
		wheel_width = max(1.0, half_width * 0.22)
		wheel_height = max(1.0, length * 0.25)
		for wheel_x in (-1.0, 1.0):
			for wheel_y in (-0.5, 0.48):
				wheel_x_center = wheel_x * half_width * 0.78
				wheel_y_center = wheel_y * length
				draw_polygon((
					(wheel_x_center - wheel_width / 2.0, wheel_y_center - wheel_height / 2.0),
					(wheel_x_center + wheel_width / 2.0, wheel_y_center - wheel_height / 2.0),
					(wheel_x_center + wheel_width / 2.0, wheel_y_center + wheel_height / 2.0),
					(wheel_x_center - wheel_width / 2.0, wheel_y_center + wheel_height / 2.0),
			), "#101416")
		draw_polygon((
			(-half_width * 0.8, length),
			(-half_width, length * 0.45),
			(-half_width, -length * 0.65),
			(-half_width * 0.55, -length),
			(half_width * 0.55, -length),
			(half_width, -length * 0.65),
			(half_width, length * 0.45),
			(half_width * 0.8, length),
		), color, outline="white", width=1)
		draw_polygon((
			(-half_width * 0.65, -length * 0.35),
			(-half_width * 0.42, -length * 0.78),
			(half_width * 0.42, -length * 0.78),
			(half_width * 0.65, -length * 0.35),
		), "#b3e5fc")
		draw_polygon((
			(-half_width * 0.12, length * 0.5),
			(-half_width * 0.09, -length * 0.25),
			(half_width * 0.09, -length * 0.25),
			(half_width * 0.12, length * 0.5),
		), "#ffffff" if color == "#e53935" else "#d8e0df")
		for light_x in (-0.58, 0.58):
			light_center_x = light_x * half_width
			light_top = length * 0.58
			light_bottom = length * 0.76
			draw_polygon((
				(light_center_x - wheel_width * 0.55, light_top),
				(light_center_x + wheel_width * 0.55, light_top),
				(light_center_x + wheel_width * 0.55, light_bottom),
				(light_center_x - wheel_width * 0.55, light_bottom),
			), "#ffd08a")

	def draw_tree(x: float, y: float, size: float) -> None:
		canvas.create_rectangle(
			x - size * 0.1, y - size * 0.25,
			x + size * 0.1, y + size * 0.65,
			fill="#55483a", outline="",
		)
		for offset, width, color in (
			(0.48, 0.72, "#355344"),
			(0.2, 0.58, "#41634e"),
			(-0.08, 0.42, "#527657"),
		):
			canvas.create_polygon(
				x, y - size,
				x - size * width, y + size * offset,
				x + size * width, y + size * offset,
				fill=color, outline="",
			)

	def draw_panel(left: float, top: float, right: float, bottom: float) -> None:
		canvas.create_rectangle(
			left, top, right, bottom,
			fill="#14211c", outline="#3a5145", width=1,
		)
		canvas.create_rectangle(
			left, top, left + 3, bottom,
			fill="#b5d779", outline="",
		)

	def draw_frame() -> None:
		nonlocal episode, crash_message, restart_at, last_frame
		now = time.perf_counter()
		dt = max(0.001, min(now - last_frame, 0.1))
		last_frame = now
		canvas_width = max(canvas.winfo_width(), 640)
		canvas_height = max(canvas.winfo_height(), 480)
		scale = min(canvas_width / 100.0, canvas_height / 75.0)
		canvas_center = canvas_width / 2.0
		car_y = canvas_height * 0.775

		if crash_message is not None:
			if now >= restart_at:
				simulation.reset()
				episode += 1
				crash_message = None
		else:
			action, log_probability = driver.choose_action(simulation.observe())
			result = simulation.step(action, dt)
			driver.record(log_probability, result.reward)
			if result.done:
				driver.finish_episode()
				recent_runs.append((simulation.player.distance, simulation.steps))
				crash_message = result.crash_reason or "episode complete"
				restart_at = now + 1.0

		player = simulation.player
		chunks = track.update(max(0.0, player.distance - 35.0))
		# Anchor the camera to the road, not the AI's lateral offset.
		road_center_x = track.center_x_at(player.distance)
		visible_start = player.distance - (canvas_height - car_y) / scale
		visible_end = player.distance + car_y / scale
		road_points = []
		for chunk in chunks:
			for point, width in zip(chunk.points.tolist(), chunk.width.tolist()):
				x, point_distance = point
				if visible_start <= point_distance <= visible_end:
					y = car_y - (point_distance - player.distance) * scale
					road_points.append((x, point_distance, width, y))

		canvas.delete("all")
		canvas.create_rectangle(
			0, 0, canvas_width, canvas_height,
			fill="#20352c", outline="",
		)
		first_tree_distance = max(0, int(visible_start // 18.0) * 18)
		for tree_distance in range(first_tree_distance, int(visible_end) + 18, 18):
			if tree_distance < 0:
				continue
			tree_y = car_y - (tree_distance - player.distance) * scale
			if tree_y < 66 or tree_y > canvas_height + 30:
				continue
			tree_scale = max(
				0.55,
				min(1.8, 1.75 - max(0.0, tree_distance - player.distance) / 75.0),
			)
			tree_size = scale * 2.2 * tree_scale
			track_x = track.center_x_at(float(tree_distance))
			for side in (-1.0, 1.0):
				tree_lateral = side * (track.track_width / 2.0 + 3.0 + tree_size / scale)
				tree_x = canvas_center + (track_x + tree_lateral - road_center_x) * scale
				if -tree_size < tree_x < canvas_width + tree_size:
					draw_tree(tree_x, tree_y, tree_size)
		if len(road_points) >= 2:
			left_edge = []
			right_edge = []
			outer_left = []
			outer_right = []
			centerline = []
			for x, point_distance, width, y in road_points:
				center_x = canvas_center + (x - road_center_x) * scale
				half_width = width * scale / 2.0
				left_edge.extend((center_x - half_width, y))
				right_edge.extend((center_x + half_width, y))
				shoulder_width = min(2.6 * scale, 24.0)
				outer_left.extend((center_x - half_width - shoulder_width, y))
				outer_right.extend((center_x + half_width + shoulder_width, y))
				centerline.append((center_x, y, point_distance, half_width))

			canvas.create_polygon(
				*(outer_left + list(reversed(left_edge))),
				fill="#283432", outline="",
			)
			canvas.create_polygon(
				*(right_edge + list(reversed(outer_right))),
				fill="#283432", outline="",
			)
			canvas.create_polygon(
				*(left_edge + list(reversed(right_edge))),
				fill="#394149",
				outline="",
			)
			for start, end in zip(centerline, centerline[1:]):
				if int(((start[2] + end[2]) / 2.0) // 8) % 2 == 0:
					curb_color = "#e78369"
				else:
					curb_color = "#e2e5d6"
				curb_width = min(0.42 * scale, 5.0)
				canvas.create_polygon(
					start[0] - start[3] - curb_width, start[1],
					start[0] - start[3], start[1],
					end[0] - end[3], end[1],
					end[0] - end[3] - curb_width, end[1],
					fill=curb_color, outline="",
				)
				canvas.create_polygon(
					start[0] + start[3], start[1],
					start[0] + start[3] + curb_width, start[1],
					end[0] + end[3] + curb_width, end[1],
					end[0] + end[3], end[1],
					fill=curb_color, outline="",
				)
			canvas.create_line(*left_edge, fill="#d9e69c", width=1)
			canvas.create_line(*right_edge, fill="#d9e69c", width=1)
			for start, end in zip(centerline, centerline[1:]):
				if int(((start[2] + end[2]) / 2.0) // 16) % 2 == 0:
					canvas.create_line(
						start[0], start[1], end[0], end[1],
						fill="#d9ded8", width=2,
					)

		for traffic_car in simulation.traffic:
			if visible_start <= traffic_car.distance <= visible_end:
				traffic_y = car_y - (traffic_car.distance - player.distance) * scale
				color = ("#74b8d1", "#e3bd69", "#d48670")[traffic_car.car_id % 3]
				draw_car(
					traffic_car.distance, traffic_car.lateral, traffic_y, color,
					scale, canvas_center, road_center_x,
				)
		draw_car(
			player.distance, player.lateral, car_y, "#e53935",
			scale, canvas_center, road_center_x,
			heading=0.65 * math.atan2(
				simulation.player_lateral_velocity,
				max(simulation.PLAYER_MIN_SPEED, player.speed),
			),
		)

		canvas.create_rectangle(
			0, 0, canvas_width, 64, fill="#101b17", outline="",
		)
		canvas.create_line(0, 63, canvas_width, 63, fill="#496052", width=1)
		canvas.create_text(
			18, 14, anchor="nw", fill="#d7e7c6",
			font=("Segoe UI", 12, "bold"),
			text="APEX / DRIVER LAB",
		)
		canvas.create_text(
			18, 38, anchor="nw", fill="#9db1a4",
			font=("Segoe UI", 9),
			text="AUTONOMOUS TRAFFIC SIMULATION",
		)
		canvas.create_text(
			canvas_width - 18, 15, anchor="ne", fill="#d9e69c",
			font=("Segoe UI", 10, "bold"),
			text=f"TRAINING ON {'GPU' if driver.device.type == 'cuda' else 'CPU'}",
		)
		canvas.create_text(
			canvas_width - 18, 38, anchor="ne", fill="#9db1a4",
			font=("Segoe UI", 9),
			text=driver.training_status,
		)

		draw_panel(16, 82, 278, 222)
		canvas.create_text(
			30, 95, anchor="nw", fill="#9db1a4",
			font=("Segoe UI", 9, "bold"), text="CURRENT RUN",
		)
		canvas.create_text(
			30, 119, anchor="nw", fill="#e6ece3",
			font=("Consolas", 11, "bold"),
			text=f"EPISODE {episode:04d}  /  STEP {simulation.steps:04d}",
		)
		canvas.create_text(
			30, 146, anchor="nw", fill="#a9c7cc",
			font=("Segoe UI", 10), text=f"DISTANCE   {player.distance:07.1f} m",
		)
		canvas.create_text(
			30, 169, anchor="nw", fill="#d9e69c",
			font=("Segoe UI", 10, "bold"),
			text=f"SPEED      {player.speed * 3.6:05.1f} km/h",
		)
		meter_left, meter_top, meter_right = 30, 194, 260
		safe_speed_limit = simulation._safe_speed_limit()
		canvas.create_rectangle(
			meter_left, meter_top, meter_right, meter_top + 5,
			fill="#34473d", outline="",
		)
		canvas.create_rectangle(
			meter_left, meter_top,
			meter_left + (meter_right - meter_left)
			* min(1.0, player.speed / simulation.PLAYER_MAX_SPEED),
			meter_top + 5,
			fill="#e87560" if player.speed > safe_speed_limit else "#b5d779",
			outline="",
		)
		target_x = meter_left + (meter_right - meter_left) * min(
			1.0, safe_speed_limit / simulation.PLAYER_MAX_SPEED
		)
		canvas.create_line(
			target_x, meter_top - 2, target_x, meter_top + 7,
			fill="#ffffff", width=2,
		)
		canvas.create_text(
			30, 204, anchor="nw", fill="#82988a",
			font=("Segoe UI", 8),
			text=f"SAFE SPEED TARGET  {safe_speed_limit * 3.6:.0f} km/h",
		)
		if recent_runs:
			mean_distance = sum(distance for distance, _ in recent_runs) / len(recent_runs)
			mean_steps = sum(steps for _, steps in recent_runs) / len(recent_runs)
			progress_text = (
				f"LAST {len(recent_runs)} AVG {mean_distance:.0f} m / {mean_steps:.0f} frames"
			)
		else:
			progress_text = "RUN AVERAGE: collecting first episode"
		footer_top = canvas_height - 58
		draw_panel(16, footer_top, min(canvas_width - 16, 340), canvas_height - 14)
		canvas.create_text(
			30, footer_top + 12, anchor="nw", fill="#d9e69c",
			font=("Segoe UI", 9, "bold"), text=progress_text,
		)
		panel_left = max(350, canvas_width - 244)
		draw_panel(panel_left, 82, canvas_width - 16, 146)
		canvas.create_text(
			panel_left + 14, 94, anchor="nw", fill="#9db1a4",
			font=("Segoe UI", 9, "bold"), text="TRAFFIC",
		)
		canvas.create_text(
			panel_left + 14, 115, anchor="nw", fill="#e6ece3",
			font=("Segoe UI", 10), text="20 km/h  |  MIN 40 km/h",
		)
		threat_detected = simulation._collision_brake() > 0.0
		canvas.create_text(
			panel_left + 14, 132, anchor="nw",
			fill="#e87560" if threat_detected else "#9fc98d",
			font=("Segoe UI", 8, "bold"),
			text="PROXIMITY ALERT" if threat_detected else "SENSORS CLEAR",
		)
		if crash_message is not None:
			canvas.create_rectangle(
				canvas_center - 190, canvas_height * 0.28,
				canvas_center + 190, canvas_height * 0.43,
				fill="#14211c", outline="#e87560", width=2,
			)
			canvas.create_text(
				canvas_center, canvas_height * 0.32,
				fill="#e87560", font=("Segoe UI", 20, "bold"),
				text=crash_message.upper(),
			)
			canvas.create_text(
				canvas_center, canvas_height * 0.38,
				fill="white", font=("Segoe UI", 11),
				text="Policy updated — restarting run",
			)
		root.after(33, draw_frame)

	draw_frame()
	root.mainloop()


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
	try:
		sizes = tuple(int(part.strip()) for part in value.split(","))
	except ValueError as exc:
		raise argparse.ArgumentTypeError(
			"hidden sizes must be comma-separated positive integers"
		) from exc
	if not sizes or any(size <= 0 for size in sizes):
		raise argparse.ArgumentTypeError(
			"hidden sizes must be comma-separated positive integers"
		)
	return sizes


def main() -> None:
	parser = argparse.ArgumentParser(description="Procedural driving AI")
	parser.add_argument(
		"--mode",
		choices=("gui", "headless", "3d"),
		default="gui",
		help="show the game or train shared simulations without a window",
	)
	parser.add_argument(
		"--terminal",
		action="store_true",
		help="run 3D mode as terminal-only training without opening a window",
	)
	parser.add_argument(
		"--device",
		choices=("cuda", "cpu"),
		default=None,
		help="training device (headless defaults to CUDA; GUI auto-selects)",
	)
	parser.add_argument("--environments", type=int)
	parser.add_argument("--updates", type=int)
	parser.add_argument(
		"--batch-episodes",
		type=int,
		default=20,
		help="completed episodes per training batch in 3D terminal mode (default: 20)",
	)
	parser.add_argument(
		"--until-stopped",
		action="store_true",
		help="train continuously until Ctrl+C; completed updates are checkpointed",
	)
	parser.add_argument("--minibatch-size", type=int, default=4096)
	parser.add_argument("--optimization-epochs", type=int, default=4)
	parser.add_argument("--learning-rate-decay", type=float, default=0.995)
	parser.add_argument("--checkpoint", type=Path)
	parser.add_argument("--seed", type=int, default=17)
	parser.add_argument("--report-every", type=int, default=1000)
	parser.add_argument(
		"--async_reset",
		"--async-reset",
		dest="async_reset",
		action="store_true",
		help="train headless from complete auto-reset episodes in thresholded batches",
	)
	parser.add_argument("--async-episodes", type=int, default=32)
	parser.add_argument("--async-transitions", type=int, default=8192)
	parser.add_argument("--max-policy-lag", type=int, default=4)
	parser.add_argument(
		"--min_batch_episodes",
		"--min-batch-episodes",
		dest="min_batch_episodes",
		type=int,
		default=128,
		help="minimum fresh complete episodes required for an async update",
	)
	parser.add_argument(
		"--max-episode-steps",
		type=int,
		default=1800,
		help="maximum steps per episode (default: 1800)",
	)
	parser.add_argument(
		"--obs_v2",
		"--obs-v2",
		dest="obs_v2",
		action="store_true",
		help="use fixed ego-frame traffic slots and road geometry inputs (headless)",
	)
	parser.add_argument(
		"--hidden",
		type=_parse_hidden_sizes,
		default=(256, 256),
		metavar="SIZES",
		help="comma-separated tanh hidden-layer widths (default: 256,256)",
	)
	parser.add_argument(
		"--traffic_density",
		"--traffic-density",
		dest="traffic_density",
		type=int,
		default=14,
		help="maximum traffic cars per simulation; 0 disables traffic",
	)
	parser.add_argument(
		"--traffic_speed",
		"--traffic-speed",
		dest="traffic_speed",
		type=float,
		default=20.0,
		help="scripted traffic speed in km/h (default: 20)",
	)
	parser.add_argument(
		"--ai_min_speed",
		"--ai-min-speed",
		dest="ai_min_speed",
		type=float,
		default=40.0,
		help="AI minimum speed in km/h (default: 40)",
	)
	parser.add_argument(
		"--curriculum",
		action="store_true",
		help="increase traffic density based on recent mean completion distance",
	)
	args = parser.parse_args()

	if args.mode == "3d" and not args.terminal:
		parser.error("--mode 3d requires --terminal")
	if args.terminal and args.mode != "3d":
		parser.error("--terminal can only be used with --mode 3d")
	if args.mode == "3d" and args.batch_episodes <= 0:
		parser.error("--batch-episodes must be positive")

	if args.mode == "gui":
		if (
			args.obs_v2
			or args.hidden != (256, 256)
			or args.traffic_density != 14
			or args.traffic_speed != 20.0
			or args.ai_min_speed != 40.0
			or args.curriculum
		):
			parser.error(
				"observation, network, and traffic options are currently "
				"supported only with --mode headless"
			)
		if args.device == "cuda" and not torch.cuda.is_available():
			raise RuntimeError(
				"CUDA was requested but is unavailable. Install CUDA-enabled "
				"PyTorch and verify the NVIDIA driver."
			)
		run_gui(device=args.device)
	else:
		from car_training import train_headless

		environments = args.environments
		if environments is None:
			environments = 32 if args.mode == "3d" else 1000
		updates = args.updates
		if updates is None and not args.until_stopped:
			updates = 100 if args.mode == "3d" else 1000
		checkpoint_path = args.checkpoint
		if args.mode == "3d" and checkpoint_path is None:
			checkpoint_path = Path(__file__).with_name(
				"3D_world_checkpoint_v1.pt"
			)
		if args.mode == "3d":
			print(
				f"3D terminal training: {updates or 'until stopped'} batches, "
				f"{args.batch_episodes} completed episodes per batch; "
				f"checkpoint family {checkpoint_path}",
				flush=True,
			)
		try:
			train_headless(
				environments=environments,
				updates=None if args.until_stopped else updates,
				device=args.device or "cuda",
				minibatch_size=args.minibatch_size,
				optimization_epochs=args.optimization_epochs,
				learning_rate_decay=args.learning_rate_decay,
				checkpoint_path=checkpoint_path,
				seed=args.seed,
				report_every=args.report_every,
				async_reset=args.async_reset or args.mode == "3d",
				async_episodes=(
					args.batch_episodes if args.mode == "3d"
					else args.async_episodes
				),
				async_transitions=args.async_transitions,
				max_policy_lag=args.max_policy_lag,
				min_batch_episodes=(
					args.batch_episodes if args.mode == "3d"
					else args.min_batch_episodes
				),
				max_episode_steps=args.max_episode_steps,
				obs_v2=args.obs_v2,
				hidden_sizes=args.hidden,
				traffic_density=args.traffic_density,
				traffic_speed_kmh=args.traffic_speed,
				ai_min_speed_kmh=args.ai_min_speed,
				curriculum=args.curriculum,
				batch_by_episodes_only=args.mode == "3d",
			)
		except KeyboardInterrupt:
			print("Training stopped by user. Latest completed update is checkpointed.")


if __name__ == "__main__":
	main()
