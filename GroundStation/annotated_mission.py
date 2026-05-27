from __future__ import annotations

import argparse
import functools
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import math

import cv2
import os
import contextlib

@contextlib.contextmanager
def _silence_stderr():
	"""
	Redirect stderr at the OS fd level to suppress C library output
	(FFmpeg/OpenH264 codec probe messages).
	Works on both Windows and Linux.
	"""
	devnull_fd = os.open(os.devnull, os.O_WRONLY)
	old_stderr_fd = os.dup(2)
	os.dup2(devnull_fd, 2)
	os.close(devnull_fd)
	try:
		yield
	finally:
		os.dup2(old_stderr_fd, 2)
		os.close(old_stderr_fd)


# Output resolution scale factor.
# 1.0 = full 1920x1080 (slow), 0.5 = 960x540 (~4x faster, good for review)
OUTPUT_SCALE: float = 0.5


# ----------------------------
# Data models
# ----------------------------

@dataclass
class MissionMeta:
	main_size: tuple[int, int]   # (w, h)
	lores_size: tuple[int, int]  # (w, h)
	fps: float
	inference_period: float


@dataclass
class DetectionSample:
	sample_id: int
	capture_ts_unix: float
	segment_t_sec: float
	estimated_main_frame_idx: int
	inference_ms: float
	detections: list[dict[str, Any]]


@dataclass
class TelemetrySample:
	timestamp: float
	altitude_m: float | None
	mode: str | None
	link_alive: bool | None


# ----------------------------
# Mirrored zone parameters from decision.py
# ----------------------------

REF_ALTITUDE_M = 10.0

ZONE2_REF = dict(cx=0.50, cy=0.58, top_w=0.18, bot_w=0.28, h=0.28)
ZONE1_REF = dict(cx=0.50, cy=0.58, top_w=0.34, bot_w=0.50, h=0.46)

ALT_SCALE_MIN = 0.25
ALT_SCALE_MAX = 2.50

ZONE_DRAW_ALPHA = 0.18


# ----------------------------
# IO helpers
# ----------------------------

def load_json(path: Path) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			line = line.strip()

			if not line:
				continue

			try:
				row = json.loads(
					line,
					parse_constant=lambda x: (_ for _ in ()).throw(
						ValueError(f"Invalid JSON constant: {x}")
					),
				)
			except (json.JSONDecodeError, ValueError) as e:
				print(f"[warning] skipping invalid JSONL line {line_no} in {path.name}: {e}")
				continue

			if not isinstance(row, dict):
				print(f"[warning] skipping non-object JSONL line {line_no} in {path.name}")
				continue

			yield row


def load_mission_meta(path: Path) -> MissionMeta:
	payload = load_json(path)

	try:
		main_size = tuple(payload["main_size"])
		lores_size = tuple(payload["lores_size"])
		fps = float(payload["fps"])
		inference_period = float(payload["inference_period"])
	except KeyError as e:
		raise ValueError(f"Missing field in mission_meta.json: {e}") from e

	if len(main_size) != 2 or len(lores_size) != 2:
		raise ValueError("main_size and lores_size must each have 2 entries")

	return MissionMeta(
		main_size=(int(main_size[0]), int(main_size[1])),
		lores_size=(int(lores_size[0]), int(lores_size[1])),
		fps=fps,
		inference_period=inference_period,
	)


def _finite_float(value: Any, field_name: str) -> float:
	value = float(value)

	if not math.isfinite(value):
		raise ValueError(f"{field_name} is not finite: {value}")

	return value


def _finite_int(value: Any, field_name: str) -> int:
	value_float = _finite_float(value, field_name)
	return int(value_float)

def load_detection_samples(path: Path) -> list[DetectionSample]:
	samples: list[DetectionSample] = []

	for row_no, row in enumerate(iter_jsonl(path), start=1):
		try:
			sample = DetectionSample(
				sample_id=_finite_int(row["sample_id"], "sample_id"),
				capture_ts_unix=_finite_float(row["capture_ts_unix"], "capture_ts_unix"),
				segment_t_sec=_finite_float(row["segment_t_sec"], "segment_t_sec"),
				estimated_main_frame_idx=_finite_int(
					row["estimated_main_frame_idx"],
					"estimated_main_frame_idx",
				),
				inference_ms=_finite_float(row["inference_ms"], "inference_ms"),
				detections=list(row.get("detections", [])),
			)
		except (KeyError, TypeError, ValueError) as e:
			print(f"[warning] skipping bad detection row {row_no} in {path.name}: {e}")
			continue

		samples.append(sample)

	samples.sort(key=lambda s: (s.estimated_main_frame_idx, s.segment_t_sec, s.sample_id))
	return samples


def load_telemetry_samples(path: Path) -> list[TelemetrySample]:
	samples: list[TelemetrySample] = []

	if not path.exists():
		return samples

	for row_no, row in enumerate(iter_jsonl(path), start=1):
		try:
			altitude = row.get("altitude_m")

			sample = TelemetrySample(
				timestamp=_finite_float(row["timestamp"], "timestamp"),
				altitude_m=(
					_finite_float(altitude, "altitude_m")
					if altitude is not None
					else None
				),
				mode=row.get("mode"),
				link_alive=row.get("link_alive"),
			)
		except (KeyError, TypeError, ValueError) as e:
			print(f"[warning] skipping bad telemetry row {row_no} in {path.name}: {e}")
			continue

		samples.append(sample)

	samples.sort(key=lambda s: s.timestamp)
	return samples

# ----------------------------
# Video path resolution
# ----------------------------

def find_matching_video(video_dir: Path, segment_stem: str) -> Path:
	candidates = [
		video_dir / f"{segment_stem}.mp4",
		video_dir / f"{segment_stem}.mkv",
		video_dir / f"{segment_stem}.avi",
		video_dir / f"{segment_stem}.mov",
		video_dir / f"{segment_stem}.h264",
	]

	for p in candidates:
		if p.exists():
			return p

	all_matches = sorted(video_dir.glob(f"{segment_stem}.*"))
	if all_matches:
		return all_matches[0]

	raise FileNotFoundError(f"No video file found for {segment_stem} inside {video_dir}")


# ----------------------------
# Box mapping
# ----------------------------

def clip_box(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int] | None:
	x1, y1, x2, y2 = box

	x1 = max(0.0, min(w - 1.0, x1))
	y1 = max(0.0, min(h - 1.0, y1))
	x2 = max(0.0, min(w - 1.0, x2))
	y2 = max(0.0, min(h - 1.0, y2))

	if x2 <= x1 or y2 <= y1:
		return None

	return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def map_bbox_stretch(
	bbox: list[float],
	lores_size: tuple[int, int],
	main_size: tuple[int, int],
) -> tuple[float, float, float, float]:
	lx, ly = lores_size
	mx, my = main_size

	sx = mx / lx
	sy = my / ly

	x1, y1, x2, y2 = bbox
	return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def map_bbox_letterbox(
	bbox: list[float],
	lores_size: tuple[int, int],
	main_size: tuple[int, int],
) -> tuple[float, float, float, float]:
	lw, lh = lores_size
	mw, mh = main_size

	scale = min(lw / mw, lh / mh)
	resized_w = mw * scale
	resized_h = mh * scale

	pad_x = (lw - resized_w) / 2.0
	pad_y = (lh - resized_h) / 2.0

	x1, y1, x2, y2 = bbox
	return (
		(x1 - pad_x) / scale,
		(y1 - pad_y) / scale,
		(x2 - pad_x) / scale,
		(y2 - pad_y) / scale,
	)


def map_bbox_center_crop(
	bbox: list[float],
	lores_size: tuple[int, int],
	main_size: tuple[int, int],
) -> tuple[float, float, float, float]:
	lw, lh = lores_size
	mw, mh = main_size

	scale = max(lw / mw, lh / mh)
	scaled_w = mw * scale
	scaled_h = mh * scale

	crop_x = (scaled_w - lw) / 2.0
	crop_y = (scaled_h - lh) / 2.0

	x1, y1, x2, y2 = bbox
	return (
		(x1 + crop_x) / scale,
		(y1 + crop_y) / scale,
		(x2 + crop_x) / scale,
		(y2 + crop_y) / scale,
	)


def resolve_bbox_mapper(
	mode: str,
	lores_size: tuple[int, int],
	main_size: tuple[int, int],
):
	if mode == "stretch":
		return map_bbox_stretch
	if mode == "letterbox":
		return map_bbox_letterbox
	if mode == "center_crop":
		return map_bbox_center_crop

	if mode != "auto":
		raise ValueError(f"Unsupported bbox map mode: {mode}")

	lores_ar = lores_size[0] / lores_size[1]
	main_ar = main_size[0] / main_size[1]

	if abs(lores_ar - main_ar) < 1e-6:
		return map_bbox_stretch

	return map_bbox_center_crop


# ----------------------------
# Zone geometry mirroring decision.py
# ----------------------------

def _scale_zone(ref: dict[str, float], alt_m: float) -> dict[str, float]:
	raw_scale = REF_ALTITUDE_M / max(alt_m, 0.5)
	scale = max(ALT_SCALE_MIN, min(ALT_SCALE_MAX, raw_scale))
	return {
		"cx": ref["cx"],
		"cy": ref["cy"],
		"top_w": min(0.95, ref["top_w"] * scale),
		"bot_w": min(0.95, ref["bot_w"] * scale),
		"h": min(0.90, ref["h"] * scale),
	}


def _trapezoid_vertices(spec: dict[str, float]) -> list[tuple[float, float]]:
	cx, cy = spec["cx"], spec["cy"]
	top_w = spec["top_w"]
	bot_w = spec["bot_w"]
	h = spec["h"]
	half_h = h / 2.0
	return [
		(cx - top_w / 2, cy - half_h),
		(cx + top_w / 2, cy - half_h),
		(cx + bot_w / 2, cy + half_h),
		(cx - bot_w / 2, cy + half_h),
	]


def zone_vertices_lores(alt_m: float, lores_size: tuple[int, int]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
	lw, lh = lores_size

	z1 = _trapezoid_vertices(_scale_zone(ZONE1_REF, alt_m))
	z2 = _trapezoid_vertices(_scale_zone(ZONE2_REF, alt_m))

	z1_px = [(x * lw, y * lh) for x, y in z1]
	z2_px = [(x * lw, y * lh) for x, y in z2]
	return z1_px, z2_px


def map_point(
	x: float,
	y: float,
	lores_size: tuple[int, int],
	main_size: tuple[int, int],
	bbox_map_mode: str,
) -> tuple[float, float]:
	mapper = resolve_bbox_mapper(bbox_map_mode, lores_size, main_size)
	x1, y1, _, _ = mapper([x, y, x, y], lores_size, main_size)
	return x1, y1


def map_polygon(
	pts: list[tuple[float, float]],
	lores_size: tuple[int, int],
	main_size: tuple[int, int],
	mapper,
) -> list[tuple[int, int]]:
	mapped = []	
	for x, y in pts:
		x1, y1, _, _ = mapper([x, y, x, y], lores_size, main_size)
		mapped.append((int(round(x1)), int(round(y1))))
	return mapped

# ----------------------------
# Drawing
# ----------------------------

def get_class_names(path: Path | None) -> dict[int, str]:
	if path is None:
		return {}

	payload = load_json(path)
	names_raw = payload.get("names", {})
	return {int(k): str(v) for k, v in names_raw.items()}


def draw_label(frame, text: str, x: int, y: int) -> None:
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.6
	thickness = 2

	(tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
	y1 = max(0, y - th - 8)
	y2 = max(0, y)
	x2 = x + tw + 6

	cv2.rectangle(frame, (x, y1), (x2, y2), (0, 255, 0), -1)
	cv2.putText(
		frame,
		text,
		(x + 3, max(12, y - 4)),
		font,
		scale,
		(0, 0, 0),
		thickness=1,
		lineType=cv2.LINE_AA,
	)


def draw_detections_on_main(
	frame,
	detections: list[dict[str, Any]],
	lores_size: tuple[int, int],
	main_size: tuple[int, int],
	class_names: dict[int, str],
	bbox_map_mode: str,
) -> Any:
	mapper = resolve_bbox_mapper(bbox_map_mode, lores_size, main_size)
	mw, mh = main_size

	for det in detections:
		bbox = det.get("bbox")
		if not bbox or len(bbox) != 4:
			continue

		mapped = mapper(bbox, lores_size, main_size)
		clipped = clip_box(mapped, mw, mh)
		if clipped is None:
			continue

		x1, y1, x2, y2 = clipped
		score = float(det.get("score", 0.0))
		cls = int(det.get("class", -1))
		name = class_names.get(cls, str(cls))
		label = f"{name} {score:.2f}"

		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		draw_label(frame, label, x1, y1)

	return frame


def draw_zones_on_main(
	frame,
	alt_m: float | None,
	lores_size: tuple[int, int],
	main_size: tuple[int, int],
	bbox_map_mode: str,
) -> Any:
	if alt_m is None:
		return frame

	mapper = resolve_bbox_mapper(bbox_map_mode, lores_size, main_size)
	z1_lores, z2_lores = zone_vertices_lores(alt_m, lores_size)
	z1_main = map_polygon(z1_lores, lores_size, main_size, mapper)
	z2_main = map_polygon(z2_lores, lores_size, main_size, mapper)

	out = frame.copy()
	overlay = out.copy()

	cv2.fillPoly(overlay, [np.array(z1_main, dtype="int32")], (0, 255, 255))
	cv2.fillPoly(overlay, [np.array(z2_main, dtype="int32")], (0, 0, 255))
	cv2.addWeighted(overlay, ZONE_DRAW_ALPHA, out, 1.0 - ZONE_DRAW_ALPHA, 0, out)  # writes into out

	cv2.polylines(out, [np.array(z1_main, dtype="int32")], True, (0, 215, 255), 2)
	cv2.polylines(out, [np.array(z2_main, dtype="int32")], True, (0, 0, 255), 2)

	if z1_main:
		x1, y1 = z1_main[0]
		cv2.putText(out, "Zone 1", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
	if z2_main:
		x2, y2 = z2_main[0]
		cv2.putText(out, "Zone 2", (x2, max(20, y2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

	return out


def draw_overlay_info(
	frame,
	frame_idx: int,
	sample: DetectionSample | None,
	telemetry: TelemetrySample | None,
) -> None:
	lines = [f"frame={frame_idx:06d}"]

	if sample is not None:
		lines.append(f"sample_id={sample.sample_id}")
		lines.append(f"segment_t={sample.segment_t_sec:.3f}s")
		lines.append(f"infer_ms={sample.inference_ms:.1f}")

	if telemetry is not None:
		if telemetry.altitude_m is not None:
			lines.append(f"alt={telemetry.altitude_m:.2f}m")
		if telemetry.mode is not None:
			lines.append(f"mode={telemetry.mode}")

	y = 24
	for line in lines:
		cv2.putText(
			frame,
			line,
			(10, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.65,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)
		y += 24


# ----------------------------
# Reconstruction logic
# ----------------------------

def open_video_capture(video_path: Path) -> cv2.VideoCapture:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")
	return cap


def make_video_writer(
	output_path: Path,
	fps: float,
	frame_size: tuple[int, int],
) -> cv2.VideoWriter:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	for codec in ("avc1", "H264", "X264", "mp4v"):
		fourcc = cv2.VideoWriter_fourcc(*codec)
		with _silence_stderr():
			writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
		if writer.isOpened():
			return writer
		writer.release()

	raise RuntimeError(f"Could not open video writer with any codec for: {output_path}")


def reconstruct_segment(
	video_path: Path,
	detections_path: Path,
	telemetry_samples: list[TelemetrySample],
	mission_meta: MissionMeta,
	output_path: Path,
	class_names: dict[int, str],
	bbox_map_mode: str,
	show_overlay_info: bool,
	draw_zones: bool,
) -> None:
	samples = load_detection_samples(detections_path)
	cap = open_video_capture(video_path)

	try:
		actual_fps = cap.get(cv2.CAP_PROP_FPS)
		fps = actual_fps if actual_fps and actual_fps > 1e-6 else mission_meta.fps

		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or mission_meta.main_size[0]
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or mission_meta.main_size[1]

		out_w = max(1, int(round(width  * OUTPUT_SCALE)))
		out_h = max(1, int(round(height * OUTPUT_SCALE)))
		writer = make_video_writer(output_path, fps, (out_w, out_h))
		try:
			frame_idx = 0
			sample_idx = 0
			telemetry_idx = 0
			active_sample: DetectionSample | None = None
			active_telemetry: TelemetrySample | None = None

			while True:
				ok, frame = cap.read()
				if not ok:
					break

				frame_time = frame_idx / fps if fps > 0 else 0.0

				while sample_idx < len(samples) and samples[sample_idx].estimated_main_frame_idx <= frame_idx:
					active_sample = samples[sample_idx]
					sample_idx += 1

				if active_sample is not None:
					current_abs_time = active_sample.capture_ts_unix + max(0.0, frame_time - active_sample.segment_t_sec)
				else:
					current_abs_time = None

				if current_abs_time is not None:
					while telemetry_idx < len(telemetry_samples) and telemetry_samples[telemetry_idx].timestamp <= current_abs_time:
						active_telemetry = telemetry_samples[telemetry_idx]
						telemetry_idx += 1

				if draw_zones and active_telemetry is not None:
					frame = draw_zones_on_main(
						frame=frame,
						alt_m=active_telemetry.altitude_m,
						lores_size=mission_meta.lores_size,
						main_size=(width, height),
						bbox_map_mode=bbox_map_mode,
					)

				if active_sample is not None and active_sample.detections:
					frame = draw_detections_on_main(
						frame=frame,
						detections=active_sample.detections,
						lores_size=mission_meta.lores_size,
						main_size=(width, height),
						class_names=class_names,
						bbox_map_mode=bbox_map_mode,
					)

				if show_overlay_info:
					draw_overlay_info(frame, frame_idx, active_sample, active_telemetry)

				if OUTPUT_SCALE != 1.0:
					frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
				writer.write(frame)
				frame_idx += 1
		finally:
			writer.release()
	finally:
		cap.release()



# ----------------------------
# Module-level segment worker (must be module-level for Windows pickling)
# ----------------------------

def _process_segment(
	det_path: Path,
	video_dir: Path,
	output_dir: Path,
	telemetry_samples: list,
	mission_meta,
	class_names: dict,
	bbox_map_mode: str,
	show_overlay_info: bool,
	draw_zones: bool,
) -> None:
	segment_stem = extract_segment_stem(det_path)
	video_path   = find_matching_video(video_dir, segment_stem)
	output_path  = output_dir / f"{segment_stem}.annotated.mp4"
	reconstruct_segment(
		video_path=video_path,
		detections_path=det_path,
		telemetry_samples=telemetry_samples,
		mission_meta=mission_meta,
		output_path=output_path,
		class_names=class_names,
		bbox_map_mode=bbox_map_mode,
		show_overlay_info=show_overlay_info,
		draw_zones=draw_zones,
	)


# ----------------------------
# Mission-level orchestration
# ----------------------------

def iter_detection_logs(log_dir: Path) -> list[Path]:
	return sorted(log_dir.glob("segment_*.detections.jsonl"))


def extract_segment_stem(detections_path: Path) -> str:
	name = detections_path.name
	suffix = ".detections.jsonl"
	if not name.endswith(suffix):
		raise ValueError(f"Unexpected detection log filename: {name}")
	return name[:-len(suffix)]


def reconstruct_mission(
	mission_root: Path,
	output_dir: Path | None,
	class_names_path: Path | None,
	bbox_map_mode: str,
	show_overlay_info: bool,
	draw_zones: bool,
) -> None:
	log_dir = mission_root / "logs"
	video_dir = mission_root / "video"

	mission_meta_path = log_dir / "mission_meta.json"
	telemetry_path = log_dir / "telemetry.jsonl"

	if not mission_meta_path.exists():
		raise FileNotFoundError(f"Missing mission meta: {mission_meta_path}")

	mission_meta = load_mission_meta(mission_meta_path)
	class_names = get_class_names(class_names_path)
	telemetry_samples = load_telemetry_samples(telemetry_path)

	detections_logs = iter_detection_logs(log_dir)
	if not detections_logs:
		raise FileNotFoundError(f"No segment_*.detections.jsonl files found in {log_dir}")

	if output_dir is None:
		output_dir = mission_root / "annotated"
	else : 
		output_dir = mission_root / output_dir

	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"[mission] root={mission_root}")
	print(f"[mission] output_dir={output_dir}")
	print(f"[mission] bbox_map_mode={bbox_map_mode}")
	print(f"[mission] logs_found={len(detections_logs)}")
	print(f"[mission] telemetry_samples={len(telemetry_samples)}")

	worker = functools.partial(
		_process_segment,
		video_dir=video_dir,
		output_dir=output_dir,
		telemetry_samples=telemetry_samples,
		mission_meta=mission_meta,
		class_names=class_names,
		bbox_map_mode=bbox_map_mode,
		show_overlay_info=show_overlay_info,
		draw_zones=draw_zones,
	)

	if len(detections_logs) == 1:
		print(f"[mission] processing 1 segment...")
		worker(detections_logs[0])
	else:
		print(f"[mission] processing {len(detections_logs)} segments in parallel...")
		with ProcessPoolExecutor() as pool:
			for i, _ in enumerate(pool.map(worker, detections_logs), 1):
				print(f"[mission] {i}/{len(detections_logs)} segments done")

	print("[mission] reconstruction complete.")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Reconstruct annotated mission videos from segmented detection and telemetry logs."
	)

	parser.add_argument(
		"mission_root",
		type=Path,
		help="Mission directory containing video/ and logs/.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory for annotated videos. Defaults to <mission_root>/annotated",
	)
	parser.add_argument(
		"--class-names-json",
		type=Path,
		default=None,
		help="Optional JSON file with {'names': {'0': 'person', ...}}. If omitted, class ids are shown.",
	)
	parser.add_argument(
		"--bbox-map-mode",
		choices=["auto", "stretch", "letterbox", "center_crop"],
		default="auto",
		help=(
			"How to map lores detections/zones into main video coordinates. "
			"Use 'stretch' when lores and main have the same aspect ratio. "
			"Use 'center_crop' or 'letterbox' if they differ."
		),
	)
	parser.add_argument(
		"--output-scale",
		type=float,
		default=0.5,
		help="Output resolution scale. 1.0=full res (slow), 0.5=half res (default, ~4x faster).",
	)
	parser.add_argument(
		"--no-overlay-info",
		action="store_true",
		help="Do not draw frame/sample timing info text.",
	)
	parser.add_argument(
		"--no-zones",
		action="store_true",
		help="Do not draw Zone 1 / Zone 2 overlays.",
	)

	return parser.parse_args()


def main() -> None:
	args = parse_args()

	global OUTPUT_SCALE
	OUTPUT_SCALE = float(args.output_scale)

	reconstruct_mission(
		mission_root=args.mission_root,
		output_dir=args.output_dir,
		class_names_path=args.class_names_json,
		bbox_map_mode=args.bbox_map_mode,
		show_overlay_info=not args.no_overlay_info,
		draw_zones=not args.no_zones,
	)


if __name__ == "__main__":
	main()