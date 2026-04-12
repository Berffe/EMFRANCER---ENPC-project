from dataclasses import dataclass
import time
import yaml
import cv2
import ncnn
import numpy as np
import json
from pathlib import Path
from pi_config import INF_THREADS

@dataclass
class ModelMeta:
	imgsz: tuple[int, int]
	names: dict[int, str]
	stride: int
	task: str


def load_model_meta(yaml_path: str) -> ModelMeta:
	with open(yaml_path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)

	names = {int(k): v for k, v in data["names"].items()}
	imgsz = tuple(data["imgsz"])
	stride = int(data["stride"])
	task = data["task"]

	return ModelMeta(
		imgsz=imgsz,
		names=names,
		stride=stride,
		task=task,
	)


def load_ncnn_model(param_path: str, bin_path: str, use_vulkan: bool = False) -> ncnn.Net:
	net = ncnn.Net()
	net.opt.use_vulkan_compute = use_vulkan  # False on Pi 4
	net.opt.num_threads = INF_THREADS
	net.load_param(str(param_path))
	net.load_model(str(bin_path))
	return net


def letterbox(image: np.ndarray, new_shape: tuple[int, int], color=(114, 114, 114)):
	h, w = image.shape[:2]
	new_w, new_h = new_shape

	scale = min(new_w / w, new_h / h)
	resized_w = int(round(w * scale))
	resized_h = int(round(h * scale))

	resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

	pad_w = new_w - resized_w
	pad_h = new_h - resized_h
	pad_left = pad_w // 2
	pad_right = pad_w - pad_left
	pad_top = pad_h // 2
	pad_bottom = pad_h - pad_top

	padded = cv2.copyMakeBorder(
		resized,
		pad_top,
		pad_bottom,
		pad_left,
		pad_right,
		cv2.BORDER_CONSTANT,
		value=color,
	)

	return padded, scale, pad_left, pad_top


def compute_iou(box1, box2) -> float:
	x1 = max(box1[0], box2[0])
	y1 = max(box1[1], box2[1])
	x2 = min(box1[2], box2[2])
	y2 = min(box1[3], box2[3])

	inter_w = max(0.0, x2 - x1)
	inter_h = max(0.0, y2 - y1)
	inter_area = inter_w * inter_h

	area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
	area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

	union = area1 + area2 - inter_area
	if union <= 0:
		return 0.0
	return inter_area / union


def nms(detections: list[dict], iou_thresh: float = 0.45) -> list[dict]:
	detections = sorted(detections, key=lambda d: d["score"], reverse=True)
	kept = []

	while detections:
		best = detections.pop(0)
		kept.append(best)

		remaining = []
		for det in detections:
			same_class = det["class"] == best["class"]
			overlap = compute_iou(det["bbox"], best["bbox"])
			if same_class and overlap > iou_thresh:
				continue
			remaining.append(det)

		detections = remaining

	return kept


def decode_yolo_ncnn_output(
	raw: np.ndarray,
	orig_shape: tuple[int, int, int],
	scale: float,
	pad_left: int,
	pad_top: int,
	num_classes: int,
	conf_thresh: float = 0.25,
) -> list[dict]:
	orig_h, orig_w = orig_shape[:2]

	raw = np.squeeze(raw)

	if raw.ndim == 1:
		raw = np.expand_dims(raw, axis=0)

	if raw.ndim == 2:
		rows, cols = raw.shape
		if rows in (4 + num_classes, 5 + num_classes) and cols > rows:
			raw = raw.T

	detections = []

	for row in raw:
		if row.ndim != 1:
			continue

		row_len = len(row)

		if row_len == 4 + num_classes:
			cx, cy, w, h = row[:4]
			class_scores = row[4:]

			if len(class_scores) == 0:
				continue

			cls = int(np.argmax(class_scores))
			score = float(class_scores[cls])

		elif row_len == 5 + num_classes:
			cx, cy, w, h = row[:4]
			obj_conf = float(row[4])
			class_scores = row[5:]

			if len(class_scores) == 0:
				continue

			cls = int(np.argmax(class_scores))
			score = float(obj_conf * class_scores[cls])

		else:
			continue

		if score < conf_thresh:
			continue

		x1 = cx - w / 2
		y1 = cy - h / 2
		x2 = cx + w / 2
		y2 = cy + h / 2

		x1 = (x1 - pad_left) / scale
		y1 = (y1 - pad_top) / scale
		x2 = (x2 - pad_left) / scale
		y2 = (y2 - pad_top) / scale

		x1 = max(0.0, min(orig_w - 1.0, x1))
		y1 = max(0.0, min(orig_h - 1.0, y1))
		x2 = max(0.0, min(orig_w - 1.0, x2))
		y2 = max(0.0, min(orig_h - 1.0, y2))

		if x2 <= x1 or y2 <= y1:
			continue

		detections.append({
			"bbox": [float(x1), float(y1), float(x2), float(y2)],
			"score": float(score),
			"class": int(cls),
		})

	return detections


def infer_ncnn(
	frame_bgr: np.ndarray,
	net: ncnn.Net,
	meta: ModelMeta,
	conf_thresh: float = 0.25,
	iou_thresh: float = 0.45,
	input_name: str = "in0",
	output_name: str = "out0",
) -> list[dict]:
	# mat = ncnn.Mat.from_pixels_resize(
	# 	frame_bgr,
	# 	ncnn.Mat.PixelType.PIXEL_BGR2RGB,
	# 	frame_bgr.shape[1],
	# 	frame_bgr.shape[0],
	# 	meta.imgsz[0],
	# 	meta.imgsz[1],
	# )

	img, scale, pad_left, pad_top = letterbox(frame_bgr, meta.imgsz)

	mat = ncnn.Mat.from_pixels(
		img,
		ncnn.Mat.PixelType.PIXEL_BGR2RGB,
		img.shape[1],
		img.shape[0],
	)

	norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
	mat.substract_mean_normalize([], norm_vals)

	ex = net.create_extractor()
	ex.input(input_name, mat)

	ret, out = ex.extract(output_name)
	if ret != 0:
		print(f"Failed to extract output '{output_name}', code={ret}")
		return []

	raw = np.asarray(out)

	detections = decode_yolo_ncnn_output(
		raw=raw,
		orig_shape=frame_bgr.shape,
		scale=scale,
		pad_left=pad_left,
		pad_top=pad_top,
		num_classes=len(meta.names),
		conf_thresh=conf_thresh,
	)

	detections = nms(detections, iou_thresh=iou_thresh)
	return detections


def draw_detections(
	frame_bgr: np.ndarray,
	detections: list[dict],
	class_names: dict[int, str],
) -> np.ndarray:
	annotated = frame_bgr.copy()

	for det in detections:
		x1, y1, x2, y2 = map(int, det["bbox"])
		score = det["score"]
		cls = det["class"]
		label_name = class_names.get(cls, str(cls))
		label = f"{label_name} {score:.2f}"

		cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

		(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
		text_y1 = max(0, y1 - th - 6)
		text_y2 = max(0, y1)
		cv2.rectangle(annotated, (x1, text_y1), (x1 + tw + 4, text_y2), (0, 255, 0), -1)
		cv2.putText(
			annotated,
			label,
			(x1 + 2, max(12, y1 - 4)),
			cv2.FONT_HERSHEY_SIMPLEX,
			2,
			(0, 0, 0),
			1,
			cv2.LINE_AA,
		)

	return annotated

def save_annotation(detections: list[dict], out_path: Path):
	payload = {
		"detections": [
			{
				"bbox": [float(v) for v in det["bbox"]],
				"score": float(det["score"]),
				"class": int(det["class"]),
			}
			for det in detections
		]
	}

	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)

def main():
	base_dir = Path(__file__).resolve().parent
	model_dir = base_dir / "yolov8n_ncnn_model"
	image_path = base_dir / "test.jpg"

	meta_path = model_dir / "metadata.yaml"
	param_path = model_dir / "model.ncnn.param"
	bin_path = model_dir / "model.ncnn.bin"

	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	if not meta_path.exists():
		raise FileNotFoundError(f"Metadata not found: {meta_path}")

	if not param_path.exists():
		raise FileNotFoundError(f"Param file not found: {param_path}")

	if not bin_path.exists():
		raise FileNotFoundError(f"Bin file not found: {bin_path}")

	meta = load_model_meta(str(meta_path))
	net = load_ncnn_model(str(param_path), str(bin_path), use_vulkan=False)

	frame_bgr = cv2.imread(str(image_path))
	if frame_bgr is None:
		raise RuntimeError(f"Failed to read image: {image_path}")

	t0 = time.time()
	detections = infer_ncnn(
		frame_bgr=frame_bgr,
		net=net,
		meta=meta,
		conf_thresh=0.25,
		iou_thresh=0.45,
	)
	infer_ms = (time.time() - t0) * 1000.0

	annotated = draw_detections(frame_bgr, detections, meta.names)

	annotated_path = base_dir / "test_annotated.jpg"
	json_path = base_dir / "test_detections.json"

	cv2.imwrite(str(annotated_path), annotated)
	save_annotation(detections, json_path)

	print(f"Inference time: {infer_ms:.1f} ms")
	print(f"Detections: {len(detections)}")
	print(f"Annotated image saved to: {annotated_path}")
	print(f"Detections JSON saved to: {json_path}")


if __name__ == "__main__":
	main()