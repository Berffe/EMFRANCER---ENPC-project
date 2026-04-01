import json
import queue
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from pi_config import (
	VIDEO_DIR,
	LOG_DIR,
	DEBUG_DIR,
	MAIN_SIZE,
	LORES_SIZE,
	FRAME_RATE,
	INFERENCE_PERIOD,
)

def ensure_dirs() -> None:
	VIDEO_DIR.mkdir(parents=True, exist_ok=True)
	LOG_DIR.mkdir(parents=True, exist_ok=True)
	DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, payload: dict[str, Any]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)

def write_mission_meta() -> None:
	write_json(
		LOG_DIR / "mission_meta.json",
		{
			"main_size": list(MAIN_SIZE),
			"lores_size": list(LORES_SIZE),
			"fps": FRAME_RATE,
			"inference_period": INFERENCE_PERIOD,
		},
	)

def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
	with open(path, "a", encoding="utf-8") as f:
		f.write(json.dumps(payload) + "\n")

def latest_put(q, item: Any, retries: int = 3, delay: float = 0.001) -> None:
	"""
	Best-effort 'keep only the latest item' queue push.

	Works with both queue.Queue and multiprocessing.Queue.
	If the queue remains full after a few retries, the new item is dropped
	instead of crashing the caller.
	"""
	for _ in range(retries):
		try:
			q.put_nowait(item)
			return
		except queue.Full:
			try:
				q.get_nowait()
			except (queue.Empty, Exception):
				pass
			time.sleep(delay)
		except Exception:
			# On weird queue state transitions, just give up quietly
			return

	# Final best-effort attempt; if still full, drop silently
	try:
		q.put_nowait(item)
	except Exception:
		pass

def draw_detections(
	frame_bgr: np.ndarray,
	detections: list[dict[str, Any]],
	class_names: dict[int, str],
) -> np.ndarray:
	frame = frame_bgr.copy()
	for det in detections:
		x1, y1, x2, y2 = map(int, det["bbox"])
		score = float(det["score"])
		cls   = int(det["class"])
		label = f"{class_names.get(cls, str(cls))} {score:.2f}"
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 0), -1)
		cv2.putText(frame, label, (x1 + 2, max(12, y1 - 4)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
	return frame