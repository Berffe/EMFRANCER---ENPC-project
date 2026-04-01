import json
import argparse
from pathlib import Path
from bisect import bisect_right

import cv2

from ncnn_wrapper import load_model_meta, draw_detections

def transform_bbox_stretch(bbox, src_size, dst_size):
	"""
	Scale bbox from src_size -> dst_size using independent x/y scaling.
	bbox = [x1, y1, x2, y2]
	src_size = (w_src, h_src)
	dst_size = (w_dst, h_dst)
	"""
	x1, y1, x2, y2 = bbox
	w_src, h_src = src_size
	w_dst, h_dst = dst_size

	sx = w_dst / w_src
	sy = h_dst / h_src

	return [
		x1 * sx,
		y1 * sy,
		x2 * sx,
		y2 * sy,
	]


def transform_bbox_center_crop(bbox, src_size, dst_size):
	"""
	Assume dst is a centered crop of src to match aspect ratio, then scaled.
	This is often a better approximation when replaying lores detections onto
	a main stream with a different aspect ratio.

	bbox = [x1, y1, x2, y2]
	src_size = (w_src, h_src)
	dst_size = (w_dst, h_dst)
	"""
	x1, y1, x2, y2 = bbox
	w_src, h_src = src_size
	w_dst, h_dst = dst_size

	src_ar = w_src / h_src
	dst_ar = w_dst / h_dst

	if abs(src_ar - dst_ar) < 1e-6:
		# same aspect ratio -> simple scale
		return transform_bbox_stretch(bbox, src_size, dst_size)

	if src_ar < dst_ar:
		# src is taller/narrower than dst -> crop vertically
		cropped_h = w_src / dst_ar
		y_offset = (h_src - cropped_h) / 2.0

		x1n, x2n = x1, x2
		y1n, y2n = y1 - y_offset, y2 - y_offset

		scale = w_dst / w_src
		return [
			x1n * scale,
			y1n * scale,
			x2n * scale,
			y2n * scale,
		]
	else:
		# src is wider than dst -> crop horizontally
		cropped_w = h_src * dst_ar
		x_offset = (w_src - cropped_w) / 2.0

		x1n, x2n = x1 - x_offset, x2 - x_offset
		y1n, y2n = y1, y2

		scale = h_dst / h_src
		return [
			x1n * scale,
			y1n * scale,
			x2n * scale,
			y2n * scale,
		]


def transform_detections(detections, src_size, dst_size, mode="center_crop"):
	transformed = []

	for det in detections:
		bbox = det["bbox"]

		if mode == "stretch":
			new_bbox = transform_bbox_stretch(bbox, src_size, dst_size)
		elif mode == "center_crop":
			new_bbox = transform_bbox_center_crop(bbox, src_size, dst_size)
		else:
			raise ValueError(f"Unknown transform mode: {mode}")

		transformed.append({
			"bbox": [float(v) for v in new_bbox],
			"score": float(det["score"]),
			"class": int(det["class"]),
		})

	return transformed

def load_detections_jsonl(path: Path) -> list[dict]:
	records = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))

	if not records:
		raise RuntimeError(f"No detection records found in {path}")

	records.sort(key=lambda r: r["timestamp"])
	return records


def build_timestamp_index(records: list[dict]):
	timestamps = [float(r["timestamp"]) for r in records]
	return timestamps


def find_record_for_timestamp(records: list[dict], timestamps: list[float], t: float) -> dict | None:
	"""
	Return the latest record with timestamp <= t.
	"""
	idx = bisect_right(timestamps, t) - 1
	if idx < 0:
		return None
	return records[idx]


def rebuild_annotated_video(
	video_path: Path,
	detections_path: Path,
	output_path: Path,
	class_names: dict[int, str],
	fps_override: float | None = None,
	show_overlay: bool = True,
) -> None:
	if not video_path.exists():
		raise FileNotFoundError(f"Video not found: {video_path}")

	if not detections_path.exists():
		raise FileNotFoundError(f"Detections log not found: {detections_path}")

	records = load_detections_jsonl(detections_path)
	timestamps = build_timestamp_index(records)

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	input_fps = cap.get(cv2.CAP_PROP_FPS)
	fps = fps_override if fps_override is not None else (input_fps if input_fps > 0 else 30.0)

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
	if not writer.isOpened():
		cap.release()
		raise RuntimeError(f"Could not open output video for writing: {output_path}")

	# We use the first logged timestamp as the video start reference.
	# This is the best assumption available with your current logs.
	t0_log = float(records[0]["timestamp"])

	frame_idx = 0

	while True:
		ok, frame_bgr = cap.read()
		if not ok:
			break

		frame_time = t0_log + (frame_idx / fps)
		record = find_record_for_timestamp(records, timestamps, frame_time)

		detections = []
		det_frame_id = -1
		det_timestamp = None
		infer_ms = None

		if record is not None:
			detections = record.get("detections", [])
			det_frame_id = int(record.get("frame_id", -1))
			det_timestamp = float(record.get("timestamp", 0.0))
			infer_ms = float(record.get("inference_ms", 0.0))

		src_size = (640, 480)          # current lores size used during mission
		dst_size = (width, height)     # video frame size, e.g. 1920x1080

		scaled_detections = transform_detections(
			detections,
			src_size=src_size,
			dst_size=dst_size,
			mode="center_crop",   # try "stretch" too if needed
		)

		annotated = draw_detections(frame_bgr, scaled_detections, class_names)

		if show_overlay:
			overlay_1 = f"video_frame={frame_idx} video_t={frame_idx / fps:.2f}s"
			overlay_2 = f"log_frame={det_frame_id} log_t={(det_timestamp - t0_log):.2f}s dets={len(detections)} infer_ms={infer_ms:.1f}" if record else "no log record yet"

			cv2.putText(
				annotated,
				overlay_1,
				(20, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 255),
				2,
				cv2.LINE_AA,
			)
			cv2.putText(
				annotated,
				overlay_2,
				(20, 60),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 255),
				2,
				cv2.LINE_AA,
			)

		writer.write(annotated)
		frame_idx += 1

		if frame_idx % 100 == 0:
			print(f"Processed {frame_idx}/{total_frames if total_frames > 0 else '?'} frames")

	cap.release()
	writer.release()

	print(f"Annotated video written to: {output_path}")


def main():
	parser = argparse.ArgumentParser(description="Rebuild annotated video from mission video + detections.jsonl")
	parser.add_argument("--video", type=str, required=True, help="Path to input video")
	parser.add_argument("--detections", type=str, required=True, help="Path to detections.jsonl")
	parser.add_argument("--model-dir", type=str, required=True, help="Path to NCNN model folder containing metadata.yaml")
	parser.add_argument("--output", type=str, default="annotated_output.mp4", help="Path to output annotated video")
	parser.add_argument("--fps", type=float, default=None, help="Optional FPS override")
	parser.add_argument("--no-overlay", action="store_true", help="Disable debug text overlay")

	args = parser.parse_args()

	model_dir = Path(args.model_dir)
	meta_path = model_dir / "metadata.yaml"
	if not meta_path.exists():
		raise FileNotFoundError(f"Metadata not found: {meta_path}")

	meta = load_model_meta(str(meta_path))

	rebuild_annotated_video(
		video_path=Path(args.video),
		detections_path=Path(args.detections),
		output_path=Path(args.output),
		class_names=meta.names,
		fps_override=args.fps,
		show_overlay=not args.no_overlay,
	)


if __name__ == "__main__":
	main()