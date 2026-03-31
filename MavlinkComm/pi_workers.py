# workers.py
import os
import queue
import threading
import time
from dataclasses import asdict
import cv2
from pi_config  import (SEGMENT_SECONDS, INFERENCE_PERIOD, CONF_THRESH, IOU_THRESH,
					META_PATH, PARAM_PATH, BIN_PATH, LOG_DIR,
					DISPLAY, SAVE_DEBUG_FRAME_EVERY, DEBUG_DIR)
from pi_types   import FramePacket, DetectionPacket
from pi_utils   import latest_put, write_jsonl
from ncnn_wrapper import load_model_meta, load_ncnn_model, infer_ncnn, draw_detections
from pi_camera  import CameraManager
from decision import DecisionEngine

# os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(10))
def _try_nice(value: int) -> None:
	try:
		os.nice(value)
	except PermissionError:
		print(f"[warning] could not set nice value {value}, continuing without priority adjustment")

def segment_rotator_worker(stop_event: threading.Event, camera: CameraManager) -> None:
	_try_nice(-5)
	while not stop_event.is_set():
		if stop_event.wait(SEGMENT_SECONDS):
			break
		print(f"[segment] rotated -> {camera.start_new_segment()}")

def frame_pump_worker(
	stop_event: threading.Event,
	camera: CameraManager,
	frame_queue: queue.Queue,
) -> None:
	_try_nice(-5)
	frame_id, next_infer_time = 0, 0.0
	while not stop_event.is_set():
		now = time.time()
		if now < next_infer_time:
			time.sleep(0.005)
			continue
		try:
			frame_bgr = camera.capture_lores_bgr()
		except Exception as e:
			print(f"[frame] capture error: {e}")
			time.sleep(0.05)
			continue
		latest_put(frame_queue, FramePacket(frame_id, now, frame_bgr))
		frame_id += 1
		next_infer_time = now + INFERENCE_PERIOD

def inference_worker(
	stop_event: threading.Event,
	frame_queue: queue.Queue,
	detection_queue: queue.Queue,
	debug_counter: list[int],
) -> None:
	_try_nice(10)
	meta = load_model_meta(str(META_PATH))
	net  = load_ncnn_model(str(PARAM_PATH), str(BIN_PATH), use_vulkan=False)
	detections_path = LOG_DIR / "detections.jsonl"

	while not stop_event.is_set():
		try:
			pkt: FramePacket = frame_queue.get(timeout=0.1)
		except queue.Empty:
			continue
		t0 = time.time()
		detections = infer_ncnn(pkt.frame_bgr, net, meta, CONF_THRESH, IOU_THRESH)
		infer_ms = (time.time() - t0) * 1000.0
		det_pkt = DetectionPacket(pkt.frame_id, pkt.timestamp, infer_ms, detections)
		write_jsonl(detections_path, {"frame_id": det_pkt.frame_id, "timestamp": det_pkt.timestamp,
									"inference_ms": det_pkt.inference_ms, "detections": det_pkt.detections})
		latest_put(detection_queue, det_pkt)
		print(f"[infer] frame={pkt.frame_id:06d} detections={len(detections)} infer_ms={infer_ms:.1f}")

		if SAVE_DEBUG_FRAME_EVERY > 0:
			debug_counter[0] += 1
			if debug_counter[0] % SAVE_DEBUG_FRAME_EVERY == 0:
				cv2.imwrite(str(DEBUG_DIR / f"debug_{pkt.frame_id:06d}.jpg"),
							draw_detections(pkt.frame_bgr, detections, meta.names))
		if DISPLAY:
			cv2.imshow("YOLO Live NCNN", draw_detections(pkt.frame_bgr, detections, meta.names))
			if cv2.waitKey(1) & 0xFF == ord("q"):
				stop_event.set()

def decision_worker(stop_event: threading.Event, detection_queue: queue.Queue) -> None:
	engine = DecisionEngine()
	decisions_path = LOG_DIR / "decisions.jsonl"
	while not stop_event.is_set():
		try:
			pkt: DetectionPacket = detection_queue.get(timeout=0.1)
		except queue.Empty:
			continue
		decision = engine.update(pkt)
		write_jsonl(decisions_path, asdict(decision))
		print(f"[decision] frame={decision.frame_id:06d} landing_clear={decision.landing_clear} reason={decision.reason}")