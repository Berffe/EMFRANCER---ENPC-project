# main.py
import queue
import threading
import time
import cv2
from pi_config  import DISPLAY
from pi_utils   import ensure_dirs
from pi_camera  import CameraManager
from pi_workers import (segment_rotator_worker, frame_pump_worker,
                     inference_worker, decision_worker)

def main() -> None:
    ensure_dirs()
    stop_event      = threading.Event()
    frame_queue     = queue.Queue(maxsize=1)
    detection_queue = queue.Queue(maxsize=1)
    debug_counter   = [0]

    camera = CameraManager()
    camera.start()

    threads = [
        threading.Thread(target=segment_rotator_worker, daemon=True, name="segment-rotator",
                         args=(stop_event, camera)),
        threading.Thread(target=frame_pump_worker,      daemon=True, name="frame-pump",
                         args=(stop_event, camera, frame_queue)),
        threading.Thread(target=inference_worker,       daemon=True, name="inference",
                         args=(stop_event, frame_queue, detection_queue, debug_counter)),
        threading.Thread(target=decision_worker,        daemon=True, name="decision",
                         args=(stop_event, detection_queue)),
    ]
    for t in threads:
        t.start()

    print("[main] system running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[main] stopping...")
        stop_event.set()
    finally:
        for t in threads:
            t.join(timeout=1.0)
        camera.stop()
        if DISPLAY:
            cv2.destroyAllWindows()
        print("[main] shutdown complete.")

if __name__ == "__main__":
    main()