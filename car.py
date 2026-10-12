"""On-demand procedural track geometry and a small playable driving game."""

from dataclasses import dataclass
import math
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


def run_gui() -> None:
	"""Drive along a scrolling, procedurally generated track."""
	import tkinter as tk
	import time

	root = tk.Tk()
	root.title("AI car simulation")
	root.geometry("800x600")
	root.minsize(640, 480)
	root.configure(bg="#254b35")

	canvas = tk.Canvas(root, width=800, height=600, bg="#254b35", highlightthickness=0)
	canvas.pack(fill="both", expand=True)

	track = ProceduralTrack(point_spacing=1.0, chunks_ahead=4)
	pressed_keys = set()
	distance = 0.0
	lateral_position = 0.0
	speed = 14.0
	last_frame = time.perf_counter()
	road_half_width = track.track_width / 2.0
	car_half_width = 2.0

	def on_key_press(event: tk.Event) -> None:
		pressed_keys.add(event.keysym.lower())

	def on_key_release(event: tk.Event) -> None:
		pressed_keys.discard(event.keysym.lower())

	root.bind("<KeyPress>", on_key_press)
	root.bind("<KeyRelease>", on_key_release)
	canvas.focus_set()

	def draw_frame() -> None:
		nonlocal distance, lateral_position, speed, last_frame

		now = time.perf_counter()
		dt = min(now - last_frame, 0.1)
		last_frame = now
		canvas_width = max(canvas.winfo_width(), 640)
		canvas_height = max(canvas.winfo_height(), 480)
		scale = min(canvas_width / 100.0, canvas_height / 75.0)
		canvas_center = canvas_width / 2.0
		car_y = canvas_height * 0.775

		if "left" in pressed_keys:
			lateral_position -= 9.0 * dt
		if "right" in pressed_keys:
			lateral_position += 9.0 * dt

		is_on_road = abs(lateral_position) + car_half_width < road_half_width
		if is_on_road:
			if "up" in pressed_keys:
				speed = min(32.0, speed + 16.0 * dt)
			elif "down" in pressed_keys:
				speed = max(0.0, speed - 28.0 * dt)
		else:
			speed = max(0.0, speed - 20.0 * dt)
		distance += speed * dt

		# Keep some track behind the car for the bottom of the scrolling view.
		chunks = track.update(max(0.0, distance - 35.0))
		center_at_car = track.center_x_at(distance)
		visible_start = distance - (canvas_height - car_y) / scale
		visible_end = distance + car_y / scale
		road_points = []
		for chunk in chunks:
			for point, width in zip(chunk.points.tolist(), chunk.width.tolist()):
				x, point_distance = point
				if visible_start <= point_distance <= visible_end:
					y = car_y - (point_distance - distance) * scale
					road_points.append((x, point_distance, width, y))

		canvas.delete("all")
		if len(road_points) >= 2:
			left_edge = []
			right_edge = []
			centerline = []
			for x, point_distance, width, y in road_points:
				center_x = canvas_center + (x - center_at_car) * scale
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

		car_x = canvas_center + lateral_position * scale
		car_length = 3.75 * scale
		canvas.create_polygon(
			car_x - car_half_width * scale * 0.75, car_y + car_length * 0.9,
			car_x - car_half_width * scale, car_y + car_length * 0.4,
			car_x - car_half_width * scale, car_y - car_length * 0.55,
			car_x - car_half_width * scale * 0.6, car_y - car_length,
			car_x + car_half_width * scale * 0.6, car_y - car_length,
			car_x + car_half_width * scale, car_y - car_length * 0.55,
			car_x + car_half_width * scale, car_y + car_length * 0.4,
			car_x + car_half_width * scale * 0.75, car_y + car_length * 0.9,
			fill="#e53935",
			outline="white",
			width=2,
		)
		canvas.create_polygon(
			car_x - car_half_width * scale * 0.65, car_y - car_length * 0.5,
			car_x - car_half_width * scale * 0.45, car_y - car_length * 0.82,
			car_x + car_half_width * scale * 0.45, car_y - car_length * 0.82,
			car_x + car_half_width * scale * 0.65, car_y - car_length * 0.5,
			fill="#b3e5fc",
			outline="",
		)

		canvas.create_rectangle(12, 12, 270, 92, fill="#17211c", outline="")
		canvas.create_text(
			26, 27, anchor="nw", fill="white",
			font=("Consolas", 14, "bold"),
			text=f"DISTANCE  {distance:07.1f} m",
		)
		canvas.create_text(
			26, 53, anchor="nw", fill="#b3e5fc",
			font=("Consolas", 14, "bold"),
			text=f"SPEED     {speed * 3.6:05.1f} km/h",
		)
		canvas.create_text(
			canvas_width - 26, 18, anchor="ne", fill="white",
			font=("Segoe UI", 11),
			text="← → steer    ↑ accelerate    ↓ brake",
		)
		root.after(33, draw_frame)

	draw_frame()
	root.mainloop()


if __name__ == "__main__":
	run_gui()
