from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2


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
				yield json.loads(line)
			except json.JSONDecodeError as e:
				raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {e}") from e


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


def load_detection_samples(path: Path) -> list[DetectionSample]:
	samples: list[DetectionSample] = []

	for row in iter_jsonl(path):
		samples.append(
			DetectionSample(
				sample_id=int(row["sample_id"]),
				capture_ts_unix=float(row["capture_ts_unix"]),
				segment_t_sec=float(row["segment_t_sec"]),
				estimated_main_frame_idx=int(row["estimated_main_frame_idx"]),
				inference_ms=float(row["inference_ms"]),
				detections=list(row.get("detections", [])),
			)
		)

	samples.sort(key=lambda s: (s.estimated_main_frame_idx, s.segment_t_sec, s.sample_id))
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
	"""
	Simple independent x/y scaling.
	Best when lores and main have the same aspect ratio.
	"""
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
	"""
	Assumes main was resized into lores with aspect-ratio preservation + padding.
	Inverse that transform.
	"""
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
	"""
	Assumes lores is a center crop of main after uniform scaling.
	This is often a better approximation than stretch when aspect ratios differ.
	"""
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

	# When aspect ratios differ, center_crop is often the least bad default.
	return map_bbox_center_crop


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
	out = frame.copy()
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

		cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
		draw_label(out, label, x1, y1)

	return out


def draw_overlay_info(
	frame,
	frame_idx: int,
	sample: DetectionSample | None,
) -> None:
	lines = [f"frame={frame_idx:06d}"]

	if sample is not None:
		lines.append(f"sample_id={sample.sample_id}")
		lines.append(f"segment_t={sample.segment_t_sec:.3f}s")
		lines.append(f"infer_ms={sample.inference_ms:.1f}")

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

	# mp4v is broadly available in OpenCV builds
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
	if not writer.isOpened():
		raise RuntimeError(f"Could not open video writer: {output_path}")
	return writer


def build_sample_lookup(
	samples: list[DetectionSample],
	total_frames: int,
	max_frame_gap: int = 0,
) -> dict[int, DetectionSample]:
	"""
	Maps exact target frames to detection samples.

	max_frame_gap=0:
		only draw on the estimated frame itself

	max_frame_gap>0:
		also draw the same detections on a small neighborhood around the target frame
	"""
	lookup: dict[int, DetectionSample] = {}

	for sample in samples:
		center = sample.estimated_main_frame_idx

		for fidx in range(center - max_frame_gap, center + max_frame_gap + 1):
			if 0 <= fidx < total_frames:
				current = lookup.get(fidx)
				if current is None:
					lookup[fidx] = sample
					continue

				# Prefer the sample whose estimated frame is closest.
				prev_dist = abs(current.estimated_main_frame_idx - fidx)
				new_dist = abs(sample.estimated_main_frame_idx - fidx)

				if new_dist < prev_dist:
					lookup[fidx] = sample
				elif new_dist == prev_dist and sample.sample_id > current.sample_id:
					lookup[fidx] = sample

	return lookup


def reconstruct_segment(
	video_path: Path,
	detections_path: Path,
	mission_meta: MissionMeta,
	output_path: Path,
	class_names: dict[int, str],
	bbox_map_mode: str,
	max_frame_gap: int,
	show_overlay_info: bool,
) -> None:
	samples = load_detection_samples(detections_path)
	cap = open_video_capture(video_path)

	try:
		actual_fps = cap.get(cv2.CAP_PROP_FPS)
		fps = actual_fps if actual_fps and actual_fps > 1e-6 else mission_meta.fps

		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or mission_meta.main_size[0]
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or mission_meta.main_size[1]

		writer = make_video_writer(output_path, fps, (width, height))
		try:
			frame_idx = 0
			sample_idx = 0
			active_sample: DetectionSample | None = None

			while True:
				ok, frame = cap.read()
				if not ok:
					break

				# Advance to the latest sample whose estimated frame is <= current frame.
				while sample_idx < len(samples) and samples[sample_idx].estimated_main_frame_idx <= frame_idx:
					active_sample = samples[sample_idx]
					sample_idx += 1

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
					draw_overlay_info(frame, frame_idx, active_sample)

				writer.write(frame)
				frame_idx += 1
		finally:
			writer.release()
	finally:
		cap.release()


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
	max_frame_gap: int,
	show_overlay_info: bool,
) -> None:
	log_dir = mission_root / "logs"
	video_dir = mission_root / "video"

	mission_meta_path = log_dir / "mission_meta.json"
	if not mission_meta_path.exists():
		raise FileNotFoundError(f"Missing mission meta: {mission_meta_path}")

	mission_meta = load_mission_meta(mission_meta_path)
	class_names = get_class_names(class_names_path)

	detections_logs = iter_detection_logs(log_dir)
	if not detections_logs:
		raise FileNotFoundError(f"No segment_*.detections.jsonl files found in {log_dir}")

	if output_dir is None:
		output_dir = mission_root / "annotated"

	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"[mission] root={mission_root}")
	print(f"[mission] output_dir={output_dir}")
	print(f"[mission] bbox_map_mode={bbox_map_mode}")
	print(f"[mission] logs_found={len(detections_logs)}")

	for det_path in detections_logs:
		segment_stem = extract_segment_stem(det_path)
		video_path = find_matching_video(video_dir, segment_stem)
		output_path = output_dir / f"{segment_stem}.annotated.mp4"

		print(f"[segment] detections={det_path.name}")
		print(f"[segment] video={video_path.name}")
		print(f"[segment] out={output_path.name}")

		reconstruct_segment(
			video_path=video_path,
			detections_path=det_path,
			mission_meta=mission_meta,
			output_path=output_path,
			class_names=class_names,
			bbox_map_mode=bbox_map_mode,
			max_frame_gap=max_frame_gap,
			show_overlay_info=show_overlay_info,
		)

	print("[mission] reconstruction complete.")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Reconstruct annotated mission videos from segmented detection logs."
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
			"How to map lores detections into main video coordinates. "
			"Use 'stretch' when lores and main have the same aspect ratio. "
			"Use 'center_crop' or 'letterbox' if they differ."
		),
	)
	parser.add_argument(
		"--max-frame-gap",
		type=int,
		default=0,
		help=(
			"How many neighboring frames around estimated_main_frame_idx should also receive the same detections. "
			"0 means annotate only the exact estimated frame."
		),
	)
	parser.add_argument(
		"--no-overlay-info",
		action="store_true",
		help="Do not draw frame/sample timing info text.",
	)

	return parser.parse_args()


def main() -> None:
	args = parse_args()

	reconstruct_mission(
		mission_root=args.mission_root,
		output_dir=args.output_dir,
		class_names_path=args.class_names_json,
		bbox_map_mode=args.bbox_map_mode,
		max_frame_gap=args.max_frame_gap,
		show_overlay_info=not args.no_overlay_info,
	)


if __name__ == "__main__":
	main()