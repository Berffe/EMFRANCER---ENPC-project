# yolo_test.py
"""
Offline test of the YOLO pipeline using existing frames.

- Reads frames from TEST_FRAMES_DIR.
- Runs YOLO on each frame.
- Optionally displays the annotated output.
- Saves annotated frames every SAVE_EVERY_N_FRAMES.
"""

import os
import time
from glob import glob

import cv2
from ultralytics import YOLO

# ----------------- CONFIG -----------------

MODEL_PATH = "runs/terrain/terrain-yolov8n6/weights/best.pt"  # change to your model
TEST_FRAMES_DIR = "dataset/test_frames"  # folder with .jpg/.png images

OUTPUT_DIR = "output/test_annotated_frames"
OUTPUT_DIR_2 = "output/test_annotations"
SAVE_EVERY_N_FRAMES = 1  # 1 = save every frame

DISPLAY = True  # usually True for debugging on laptop


# ----------------- HELPERS -----------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_test_frames(image_dir):
    """
    Generator yielding:
        frame_bgr, frame_idx, source_name
    where:
        frame_bgr: numpy BGR image from cv2.imread
        frame_idx: sequential index
        source_name: base filename (no extension)
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(image_dir, ext)))

    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"[TEST] No images found in {image_dir}")

    for idx, path in enumerate(paths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"[TEST] Warning: could not read {path}")
            continue
        source_name = os.path.splitext(os.path.basename(path))[0]
        yield frame, idx, source_name


def should_save(frame_idx: int) -> bool:
    if SAVE_EVERY_N_FRAMES <= 0:
        return False
    return (frame_idx % SAVE_EVERY_N_FRAMES) == 0


def save_annotated_frame(frame_bgr, frame_idx: int, source_name: str):
    """
    Save an annotated frame to OUTPUT_DIR.
    """
    ensure_dir(OUTPUT_DIR)
    filename = f"test_{frame_idx:06d}_{source_name}.jpg"
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, frame_bgr)
    print(f"[TEST] Saved frame -> {out_path}")


def results_to_lines(model_r, frame_idx: int):
    """
    Convert YOLO Results for one frame into a list of text lines.
    """
    lines = []
    if model_r.boxes is None or len(model_r.boxes) == 0:
        return lines

    boxes = model_r.boxes.xyxy.cpu().numpy()
    confs = model_r.boxes.conf.cpu().numpy()
    clss  = model_r.boxes.cls.cpu().numpy()

    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = box
        class_id = int(cls_id)
        class_name = model_r.names[class_id]
        line = f"{frame_idx} {class_name} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {conf:.3f}"
        lines.append(line)

    return lines

def save_annotation(model_r, frame_idx: int, source_name: str):
	"""
	Save an annotation to OUTPUT_DIR.
	"""
	ensure_dir(OUTPUT_DIR_2)
	filename = f"a_test_{frame_idx:06d}_{source_name}.txt"
	out_path = os.path.join(OUTPUT_DIR_2, filename)
	file = open(out_path, "w")
	lines = results_to_lines(model_r, frame_idx)
	for line in lines:
		file.write(line + '\n')
	print(f"[TEST] Saved annotations -> {out_path}")


# ----------------- MAIN -----------------

def main():
    print(f"[TEST] Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[TEST] Model loaded.")
    print(f"[TEST] Using frames from: {TEST_FRAMES_DIR}")

    start_time = time.time()
    processed = 0

    for frame_bgr, idx, source_name in iter_test_frames(TEST_FRAMES_DIR):
        # Run YOLO
        results = model(frame_bgr, verbose=False)
        r = results[0]

        annotated = r.plot()

        # Optional display
        if DISPLAY:
            cv2.imshow("YOLO Test Offline", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Save annotated frame
        if should_save(idx):
            save_annotated_frame(annotated, idx, source_name)
            save_annotation(r, idx, source_name)

        processed += 1

    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"[TEST] Finished. Processed {processed} frames in {elapsed:.2f} s.")


if __name__ == "__main__":
    main()
