# pi_workers.py
import os
import queue
import time
from dataclasses import asdict

from pi_config import (
	SEGMENT_SECONDS,
	INFERENCE_PERIOD,
	LOG_DIR,
)
from pi_types import FramePacket, DetectionPacket
from pi_utils import write_jsonl
from decision import DecisionEngine


def _try_nice(value: int) -> None:
	try:
		os.nice(value)
	except PermissionError:
		print(f"[warning] could not set nice value {value}, continuing without priority adjustment")


def segment_rotator_worker(stop_event, camera) -> None:
	_try_nice(-5)
	while not stop_event.is_set():
		if stop_event.wait(SEGMENT_SECONDS):
			break
		print(f"[segment] rotated -> {camera.start_new_segment()}")


def frame_pump_worker(stop_event, camera, frame_queue) -> None:
	import cv2

	_try_nice(-5)
	frame_id = 0
	next_infer_time = 0.0

	while not stop_event.is_set():
		now = time.time()
		if now < next_infer_time:
			time.sleep(0.005)
			continue

		try:
			lores_raw = camera.get_latest_lores_raw()
			if lores_raw is None:
				time.sleep(0.01)
				continue

			# Convert only when we actually need a frame for inference
			frame_bgr = cv2.cvtColor(lores_raw, cv2.COLOR_YUV2BGR_I420)
		except Exception as e:
			print(f"[frame] capture error: {e}")
			time.sleep(0.05)
			continue

		pkt = FramePacket(frame_id, now, frame_bgr)

		try:
			frame_queue.put_nowait(pkt)
		except queue.Full:
			pass
		except Exception as e:
			print(f"[frame] queue push error: {e}")

		frame_id += 1
		next_infer_time = now + INFERENCE_PERIOD

def inference_worker(
	stop_event,
	ready_event,
	frame_queue,
	detection_queue,
	debug_counter,
) -> None:
	import os
	import queue
	import time
	import cv2

	from pi_config import (
		META_PATH,
		PARAM_PATH,
		BIN_PATH,
		CONF_THRESH,
		IOU_THRESH,
		DISPLAY,
		SAVE_DEBUG_FRAME_EVERY,
		DEBUG_DIR,
		LOG_DIR,
	)
	from ncnn_wrapper import load_model_meta, load_ncnn_model, infer_ncnn, draw_detections
	from pi_types import DetectionPacket
	from pi_utils import write_jsonl

	_try_nice(10)

	try:
		os.sched_setaffinity(0, {2, 3})
		print("[infer] pinned to CPU cores {2, 3}")
	except Exception as e:
		print(f"[warning] could not set CPU affinity: {e}")

	try:
		meta = load_model_meta(str(META_PATH))
		net = load_ncnn_model(str(PARAM_PATH), str(BIN_PATH), use_vulkan=False)
		detections_path = LOG_DIR / "detections.jsonl"
	except Exception as e:
		print(f"[infer] startup error: {e}")
		ready_event.set()  # avoid deadlocking main on startup wait
		return

	ready_event.set()
	print("[infer] ready.")

	try:
		while not stop_event.is_set():
			try:
				pkt = frame_queue.get(timeout=0.1)
			except queue.Empty:
				continue
			except (KeyboardInterrupt, EOFError):
				break
			except Exception:
				continue

			try:
				t0 = time.time()
				detections = infer_ncnn(pkt.frame_bgr, net, meta, CONF_THRESH, IOU_THRESH)
				infer_ms = (time.time() - t0) * 1000.0
			except KeyboardInterrupt:
				break
			except EOFError:
				break
			except Exception as e:
				print(f"[infer] error on frame {getattr(pkt, 'frame_id', 'unknown')}: {e}")
				continue

			det_pkt = DetectionPacket(pkt.frame_id, pkt.timestamp, infer_ms, detections)

			write_jsonl(
				detections_path,
				{
					"frame_id": det_pkt.frame_id,
					"timestamp": det_pkt.timestamp,
					"inference_ms": det_pkt.inference_ms,
					"detections": det_pkt.detections,
				},
			)

			# freshest-only policy
			try:
				while True:
					detection_queue.get_nowait()
			except Exception:
				pass

			try:
				detection_queue.put_nowait(det_pkt)
			except Exception:
				pass

			print(
				f"[infer] frame={pkt.frame_id:06d} "
				f"detections={len(detections)} "
				f"infer_ms={infer_ms:.1f}"
			)

			if SAVE_DEBUG_FRAME_EVERY > 0:
				with debug_counter.get_lock():
					debug_counter.value += 1
					should_save = (debug_counter.value % SAVE_DEBUG_FRAME_EVERY == 0)

				if should_save:
					cv2.imwrite(
						str(DEBUG_DIR / f"debug_{pkt.frame_id:06d}.jpg"),
						draw_detections(pkt.frame_bgr, detections, meta.names),
					)

			if DISPLAY:
				cv2.imshow("YOLO Live NCNN", draw_detections(pkt.frame_bgr, detections, meta.names))
				if cv2.waitKey(1) & 0xFF == ord("q"):
					stop_event.set()
					break

	except KeyboardInterrupt:
		pass
	finally:
		print("[infer] shutdown complete.")


def decision_worker(stop_event, detection_queue) -> None:
	engine = DecisionEngine()
	decisions_path = LOG_DIR / "decisions.jsonl"

	while not stop_event.is_set():
		try:
			pkt: DetectionPacket = detection_queue.get(timeout=0.1)
		except queue.Empty:
			continue
		except Exception:
			continue

		decision = engine.update(pkt)
		write_jsonl(decisions_path, asdict(decision))
		print(
			f"[decision] frame={decision.frame_id:06d} "
			f"landing_clear={decision.landing_clear} "
			f"reason={decision.reason}"
		)