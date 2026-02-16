# yolo_live.py
"""
Real-time inference from the embarked camera (PiCamera).

- Uses camera.PiCamera to grab frames.
- Runs YOLO on each frame.
- Optionally displays the annotated output.
- Saves annotated frames every SAVE_EVERY_N_FRAMES.
"""

import os
import time

import cv2
from ultralytics import YOLO
from camera import PiCamera  # you will implement this in camera.py

# ----------------- CONFIG -----------------

MODEL_PATH = "runs/terrain/terrain-yolov8n/weights/best.pt"  # change to your model
OUTPUT_DIR = "output/live_annotated_frames"

# Save every N frames (1 = every frame, 5 = 1 out of 5, etc.)
SAVE_EVERY_N_FRAMES = 5

# Show a live window (usually False on the Pi in flight, True for bench tests)
DISPLAY = False

CAMERA_RESOLUTION = (640, 480)  # adapt to your needs


# ----------------- HELPERS -----------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def should_save(frame_idx: int) -> bool:
    if SAVE_EVERY_N_FRAMES <= 0:
        return False
    return (frame_idx % SAVE_EVERY_N_FRAMES) == 0


def save_annotated_frame(frame_bgr, frame_idx: int):
    """
    Save an annotated frame to OUTPUT_DIR.
    """
    ensure_dir(OUTPUT_DIR)
    filename = f"live_{frame_idx:06d}.jpg"
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, frame_bgr)
    print(f"[LIVE] Saved frame -> {out_path}")


# ----------------- MAIN -----------------

def main():
    print(f"[LIVE] Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[LIVE] Model loaded.")

    cam = PiCamera(size=CAMERA_RESOLUTION)
    print("[LIVE] PiCamera initialized.")

    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            # Grab frame from embarked camera
            frame_bgr = cam.get_frame()  # BGR numpy array

            # Run YOLO inference
            results = model(frame_bgr, verbose=False)
            r = results[0]

            # Annotated frame with bounding boxes, labels, etc.
            annotated = r.plot()  # BGR

            # Optional display for bench testing
            if DISPLAY:
                cv2.imshow("YOLO Live", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Save according to frequency
            if should_save(frame_idx):
                save_annotated_frame(annotated, frame_idx)

            frame_idx += 1

    finally:
        cam.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"[LIVE] Stopped. Processed {frame_idx} frames in {elapsed:.2f} s.")


if __name__ == "__main__":
    main()
