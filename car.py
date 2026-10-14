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
	root.title("Procedural Traffic AI")
	root.geometry("800x600")
	root.minsize(640, 480)
	root.configure(bg="#254b35")

	canvas = tk.Canvas(root, width=800, height=600, bg="#254b35", highlightthickness=0)
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
	) -> None:
		center_x = canvas_center + (
			track.center_x_at(distance) + lateral - road_center_x
		) * scale
		half_width = 1.1 * scale
		length = 2.0 * scale
		canvas.create_polygon(
			center_x - half_width * 0.8, y + length,
			center_x - half_width, y + length * 0.45,
			center_x - half_width, y - length * 0.65,
			center_x - half_width * 0.55, y - length,
			center_x + half_width * 0.55, y - length,
			center_x + half_width, y - length * 0.65,
			center_x + half_width, y + length * 0.45,
			center_x + half_width * 0.8, y + length,
			fill=color,
			outline="white",
			width=1,
		)
		canvas.create_polygon(
			center_x - half_width * 0.65, y - length * 0.35,
			center_x - half_width * 0.42, y - length * 0.78,
			center_x + half_width * 0.42, y - length * 0.78,
			center_x + half_width * 0.65, y - length * 0.35,
			fill="#b3e5fc",
			outline="",
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
		if len(road_points) >= 2:
			left_edge = []
			right_edge = []
			centerline = []
			for x, point_distance, width, y in road_points:
				center_x = canvas_center + (x - road_center_x) * scale
				half_width = width * scale / 2.0
				left_edge.extend((center_x - half_width, y))
				right_edge.extend((center_x + half_width, y))
				centerline.append((center_x, y, point_distance))

			canvas.create_polygon(
				*(left_edge + list(reversed(right_edge))),
				fill="#454b50",
				outline="",
			)
			canvas.create_line(*left_edge, fill="#f4d35e", width=3)
			canvas.create_line(*right_edge, fill="#f4d35e", width=3)
			for start, end in zip(centerline, centerline[1:]):
				if int(((start[2] + end[2]) / 2.0) // 16) % 2 == 0:
					canvas.create_line(
						start[0], start[1], end[0], end[1],
						fill="#f4f4f4", width=3,
					)

		for traffic_car in simulation.traffic:
			if visible_start <= traffic_car.distance <= visible_end:
				traffic_y = car_y - (traffic_car.distance - player.distance) * scale
				color = ("#42a5f5", "#ffca28", "#ab47bc")[traffic_car.car_id % 3]
				draw_car(
					traffic_car.distance, traffic_car.lateral, traffic_y, color,
					scale, canvas_center, road_center_x,
				)
		draw_car(
			player.distance, player.lateral, car_y, "#e53935",
			scale, canvas_center, road_center_x,
		)

		canvas.create_rectangle(12, 12, 380, 138, fill="#17211c", outline="")
		canvas.create_text(
			26, 27, anchor="nw", fill="white",
			font=("Consolas", 13, "bold"),
			text=f"EPISODE   {episode:04d}   STEPS {simulation.steps:04d}",
		)
		canvas.create_text(
			26, 52, anchor="nw", fill="#b3e5fc",
			font=("Consolas", 13, "bold"),
			text=f"DISTANCE  {player.distance:07.1f} m",
		)
		canvas.create_text(
			26, 77, anchor="nw", fill="#a5d6a7",
			font=("Consolas", 13, "bold"),
			text=f"SPEED     {player.speed * 3.6:05.1f} km/h",
		)
		if recent_runs:
			mean_distance = sum(distance for distance, _ in recent_runs) / len(recent_runs)
			mean_steps = sum(steps for _, steps in recent_runs) / len(recent_runs)
			progress_text = (
				f"LAST {len(recent_runs)} AVG {mean_distance:.0f} m / {mean_steps:.0f} frames"
			)
		else:
			progress_text = "RUN AVERAGE: collecting first episode"
		canvas.create_text(
			26, 103, anchor="nw", fill="#ffe082",
			font=("Consolas", 11, "bold"),
			text=progress_text,
		)
		device_name = "GPU" if driver.device.type == "cuda" else "CPU"
		canvas.create_text(
			canvas_width - 20, 18, anchor="ne", fill="white",
			font=("Segoe UI", 11),
			text="TRAFFIC 20 km/h  |  AI MIN 40 km/h",
		)
		canvas.create_text(
			canvas_width - 20, 40, anchor="ne", fill="#b3e5fc",
			font=("Segoe UI", 10),
			text=f"TRAINING ON {device_name}  |  {driver.episodes_trained} updates",
		)
		if crash_message is not None:
			canvas.create_rectangle(
				canvas_center - 190, canvas_height * 0.28,
				canvas_center + 190, canvas_height * 0.43,
				fill="#17211c", outline="#ef5350", width=2,
			)
			canvas.create_text(
				canvas_center, canvas_height * 0.32,
				fill="#ef5350", font=("Segoe UI", 20, "bold"),
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


def main() -> None:
	parser = argparse.ArgumentParser(description="Procedural driving AI")
	parser.add_argument(
		"--mode",
		choices=("gui", "headless"),
		default="gui",
		help="show the game or train shared simulations without a window",
	)
	parser.add_argument(
		"--device",
		choices=("cuda", "cpu"),
		default=None,
		help="training device (headless defaults to CUDA; GUI auto-selects)",
	)
	parser.add_argument("--environments", type=int, default=1000)
	parser.add_argument("--updates", type=int, default=1000)
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
	args = parser.parse_args()

	if args.mode == "gui":
		if args.device == "cuda" and not torch.cuda.is_available():
			raise RuntimeError(
				"CUDA was requested but is unavailable. Install CUDA-enabled "
				"PyTorch and verify the NVIDIA driver."
			)
		run_gui(device=args.device)
	else:
		from car_training import train_headless

		train_headless(
			environments=args.environments,
			updates=args.updates,
			device=args.device or "cuda",
			minibatch_size=args.minibatch_size,
			optimization_epochs=args.optimization_epochs,
			learning_rate_decay=args.learning_rate_decay,
			checkpoint_path=args.checkpoint,
			seed=args.seed,
			report_every=args.report_every,
			async_reset=args.async_reset,
			async_episodes=args.async_episodes,
			async_transitions=args.async_transitions,
			max_policy_lag=args.max_policy_lag,
			min_batch_episodes=args.min_batch_episodes,
			max_episode_steps=args.max_episode_steps,
		)


if __name__ == "__main__":
	main()
