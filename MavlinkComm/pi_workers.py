import os
import queue
import time
from dataclasses import asdict

from pi_config import (
	SEGMENT_SECONDS,
	INFERENCE_PERIOD,
	FRAME_RATE,
	LOG_DIR,
	MAVLINK_SUPPRESS_DUPLICATES,
	TELEMETRY_POLL_INTERVAL,
	LOG_DECISION,
)
from pi_types import (
	FramePacket,
	DetectionPacket,
	CommandPacket,
	VehicleStatePacket,
)
from pi_utils import write_jsonl
from decision import DecisionEngine
from mavlink_client import MAVLinkError


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

			segment_index, segment_start_ts, _ = camera.get_current_segment_info()
			if segment_index < 0:
				time.sleep(0.01)
				continue

			segment_t_sec = max(0.0, now - segment_start_ts)
			estimated_main_frame_idx = round(segment_t_sec * FRAME_RATE)

			frame_bgr = cv2.cvtColor(lores_raw, cv2.COLOR_YUV2BGR_I420)
		except Exception as e:
			print(f"[frame] capture error: {e}")
			time.sleep(0.05)
			continue

		pkt = FramePacket(
			frame_id,
			now,
			segment_index,
			segment_t_sec,
			estimated_main_frame_idx,
			frame_bgr,
		)

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
	)
	from ncnn_wrapper import load_model_meta, load_ncnn_model, infer_ncnn, draw_detections

	_try_nice(10)

	try:
		os.sched_setaffinity(0, {2, 3})
		print("[infer] pinned to CPU cores {2, 3}")
	except Exception as e:
		print(f"[warning] could not set CPU affinity: {e}")

	try:
		meta = load_model_meta(str(META_PATH))
		net = load_ncnn_model(str(PARAM_PATH), str(BIN_PATH), use_vulkan=False)
	except Exception as e:
		print(f"[infer] startup error: {e}")
		ready_event.set()
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

			det_pkt = DetectionPacket(
				pkt.frame_id,
				pkt.timestamp,
				pkt.segment_index,
				pkt.segment_t_sec,
				pkt.estimated_main_frame_idx,
				infer_ms,
				detections,
			)

			detections_path = LOG_DIR / f"segment_{pkt.segment_index:04d}.detections.jsonl"
			write_jsonl(
				detections_path,
				{
					"sample_id": det_pkt.frame_id,
					"capture_ts_unix": det_pkt.timestamp,
					"segment_t_sec": det_pkt.segment_t_sec,
					"estimated_main_frame_idx": det_pkt.estimated_main_frame_idx,
					"inference_ms": det_pkt.inference_ms,
					"detections": det_pkt.detections,
				},
			)

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
				f"segment={pkt.segment_index:04d} "
				f"main_frame={pkt.estimated_main_frame_idx:06d} "
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


def telemetry_worker(stop_event, mavlink_client, telemetry_state) -> None:
	telemetry_path = LOG_DIR / "telemetry.jsonl"

	while not stop_event.is_set():
		now = time.time()

		try:
			state = mavlink_client.get_vehicle_state()
			pkt = VehicleStatePacket(
				timestamp=now,
				link_alive=mavlink_client.is_link_alive(),
				armed=state.get("armed"),
				mode=state.get("mode"),
				altitude_m=state.get("altitude_m"),
				lat=state.get("lat"),
				lon=state.get("lon"),
				voltage_V=state.get("voltage_V"),
				gps_fix=state.get("gps_fix"),
			)
		except Exception as e:
			print(f"[telemetry] read error: {e}")
			pkt = VehicleStatePacket(
				timestamp=now,
				link_alive=False,
				armed=None,
				mode=None,
				altitude_m=None,
				lat=None,
				lon=None,
				voltage_V=None,
				gps_fix=None,
			)

		telemetry_state.update(pkt)
		write_jsonl(telemetry_path, asdict(pkt))

		if stop_event.wait(TELEMETRY_POLL_INTERVAL):
			break


def decision_worker(stop_event, detection_queue, command_queue, telemetry_state) -> None:
	engine = DecisionEngine()
	decisions_path = LOG_DIR / "decisions.jsonl"
	commands_path = LOG_DIR / "commands.jsonl"

	while not stop_event.is_set():
		try:
			pkt: DetectionPacket = detection_queue.get(timeout=0.1)
		except queue.Empty:
			continue
		except Exception:
			continue

		vehicle_state = telemetry_state.get_snapshot()
		decision = engine.update(pkt, vehicle_state=vehicle_state)
		if LOG_DECISION:
			write_jsonl(decisions_path, asdict(decision))

		command = CommandPacket(
			frame_id=decision.frame_id,
			timestamp=decision.timestamp,
			action=decision.action,
			reason=decision.reason,
			confidence=decision.confidence,
		)

		write_jsonl(commands_path, asdict(command))

		try:
			while True:
				command_queue.get_nowait()
		except Exception:
			pass

		try:
			command_queue.put_nowait(command)
		except Exception:
			pass

		mode = vehicle_state.mode if vehicle_state is not None else None
		alt = vehicle_state.altitude_m if vehicle_state is not None else None

		print(
			f"[decision] frame={decision.frame_id:06d} "
			f"landing_clear={decision.landing_clear} "
			f"action={decision.action} "
			f"reason={decision.reason} "
			f"mode={mode} alt={alt}"
		)


def command_worker(stop_event, command_queue, mavlink_client) -> None:
	last_action = None

	while not stop_event.is_set():
		try:
			pkt: CommandPacket = command_queue.get(timeout=0.1)
		except queue.Empty:
			continue
		except Exception:
			continue

		if MAVLINK_SUPPRESS_DUPLICATES and pkt.action == last_action:
			print(f"[command] suppressed duplicate action={pkt.action}")
			continue

		try:
			mavlink_client.send_action(pkt.action)
			last_action = pkt.action
			print(
				f"[command] sent action={pkt.action} "
				f"frame={pkt.frame_id:06d} "
				f"reason={pkt.reason}"
			)
		except MAVLinkError as e:
			print(f"[command] send error: {e}")
		except Exception as e:
			print(f"[command] unexpected error: {e}")