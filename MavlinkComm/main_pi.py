# main_pi.py
import multiprocessing as mp
import queue
import threading
import time
import cv2

from pi_config import DISPLAY
from pi_utils import ensure_dirs, write_mission_meta
from pi_camera import CameraManager
from pi_workers import (
	segment_rotator_worker,
	frame_pump_worker,
	inference_worker,
	decision_worker,
)


def main() -> None:
	ensure_dirs()
	write_mission_meta()

	stop_event = mp.Event()
	ready_event = mp.Event()

	frame_queue = mp.Queue(maxsize=1)
	detection_queue = mp.Queue(maxsize=1)
	debug_counter = mp.Value("i", 0)

	infer_proc = mp.Process(
		target=inference_worker,
		daemon=True,
		name="inference-process",
		args=(stop_event, ready_event, frame_queue, detection_queue, debug_counter),
	)
	infer_proc.start()

	print("[main] waiting for inference process...")
	if not ready_event.wait(timeout=15.0):
		print("[main] inference process failed to become ready in time.")
		stop_event.set()
		if infer_proc.is_alive():
			infer_proc.terminate()
			infer_proc.join(timeout=1.0)
		return

	camera = CameraManager()
	camera.start()

	threads = [
		threading.Thread(
			target=segment_rotator_worker,
			daemon=True,
			name="segment-rotator",
			args=(stop_event, camera),
		),
		threading.Thread(
			target=frame_pump_worker,
			daemon=True,
			name="frame-pump",
			args=(stop_event, camera, frame_queue),
		),
		threading.Thread(
			target=decision_worker,
			daemon=True,
			name="decision",
			args=(stop_event, detection_queue),
		),
	]

	for t in threads:
		t.start()

	print("[main] system running. Press Ctrl+C to stop.")
	try:
		while True:
			time.sleep(1.0)
	except KeyboardInterrupt:
		print("\n[main] stopping...")
	finally:
		stop_event.set()

		for t in threads:
			t.join(timeout=0.5)

		if infer_proc.is_alive():
			infer_proc.terminate()
			infer_proc.join(timeout=0.5)

		try:
			frame_queue.cancel_join_thread()
			frame_queue.close()
		except Exception:
			pass

		try:
			detection_queue.cancel_join_thread()
			detection_queue.close()
		except Exception:
			pass

		try:
			camera.stop()
		except Exception:
			pass

		if DISPLAY:
			cv2.destroyAllWindows()

		print("[main] shutdown complete.")


if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	main()