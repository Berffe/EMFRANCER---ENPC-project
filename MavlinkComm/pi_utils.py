# utils.py
import json
import queue
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from pi_config import VIDEO_DIR, LOG_DIR, DEBUG_DIR

def ensure_dirs() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

def latest_put(q: queue.Queue, item: Any) -> None:
    """Drop stale item if queue is full, then insert the new one."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)

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