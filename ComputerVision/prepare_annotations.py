from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml

ROOT = Path(__file__).resolve().parent

DATASET_FOLDERS = [
    ROOT / "Dataset01",
]

OUTPUT_ROOT = ROOT / "Dataset01_prepared"
COPY_FILES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FINAL_CLASSES: Dict[str, int] = {
    "person": 0,
    "vehicle": 1,
    "animal": 2,
    "tree": 3,
    "building": 4,
    "landing_zone": 5,
}

FINAL_CLASS_NAMES = {v: k for k, v in FINAL_CLASSES.items()}

# VisDrone-style source classes
DATASET01_SOURCE_CLASSES: Dict[int, str] = {
    0: "ignored_regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

DATASET01_TO_FINAL: Dict[str, str] = {
    "pedestrian": "person",
    "people": "person",
    "bicycle": "vehicle",
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "tricycle": "vehicle",
    "awning-tricycle": "vehicle",
    "bus": "vehicle",
    "motor": "vehicle",
}

DATASET_CONFIG = {
    "Dataset01": {
        "source_classes": DATASET01_SOURCE_CLASSES,
        "mapping": DATASET01_TO_FINAL,
        "train_folder": "train",
        "val_folder": "valid",
        "annotation_folder": "annotations",
    }
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def find_image_files(images_dir: Path) -> List[Path]:
    return sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def visdrone_to_yolo_labels(
    image_path: Path,
    annotation_path: Path,
    source_classes: Dict[int, str],
    class_mapping: Dict[str, str],
) -> List[Tuple[int, float, float, float, float]]:
    """
    Convert VisDrone annotation lines:
    x,y,w,h,score,category,truncation,occlusion
    to YOLO labels:
    class_id x_center y_center width height
    """
    if not annotation_path.exists():
        return []

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return []

    img_h, img_w = image.shape[:2]
    converted = []

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split(",")
            if len(parts) < 8:
                print(f"[WARN] Bad annotation line in {annotation_path}:{line_num}")
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                score = int(parts[4])
                category_id = int(parts[5])
                truncation = int(parts[6])
                occlusion = int(parts[7])
            except ValueError:
                print(f"[WARN] Could not parse line in {annotation_path}:{line_num}")
                continue

            # Ignore VisDrone ignored regions / invalid classes
            source_name = source_classes.get(category_id)
            if source_name is None or source_name == "ignored_regions":
                continue

            target_name = class_mapping.get(source_name)
            if target_name is None:
                continue

            target_id = FINAL_CLASSES[target_name]

            # Convert xywh (top-left based) -> YOLO normalized xywh
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            width = w / img_w
            height = h / img_h

            # Clamp to [0, 1]
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            width = min(max(width, 0.0), 1.0)
            height = min(max(height, 0.0), 1.0)

            if width <= 0 or height <= 0:
                continue

            converted.append((target_id, x_center, y_center, width, height))

    return converted


def write_yolo_labels(label_path: Path, labels: List[Tuple[int, float, float, float, float]]) -> None:
    ensure_dir(label_path.parent)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, x, y, w, h in labels:
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def prepare_output_dirs(output_root: Path) -> None:
    for split in ("train", "val"):
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)


def process_split(
    dataset_dir: Path,
    split_source_folder: str,
    split_output_name: str,
    source_classes: Dict[int, str],
    class_mapping: Dict[str, str],
    annotation_folder: str,
) -> int:
    images_dir = dataset_dir / split_source_folder / "images"
    annotations_dir = dataset_dir / split_source_folder / annotation_folder

    if not images_dir.exists():
        print(f"[WARN] Missing images directory: {images_dir}")
        return 0

    if not annotations_dir.exists():
        print(f"[WARN] Missing annotations directory: {annotations_dir}")
        return 0

    count = 0
    image_files = find_image_files(images_dir)

    print(f"[INFO] Processing {split_output_name}: {len(image_files)} images found")

    for image_path in image_files:
        annotation_path = annotations_dir / f"{image_path.stem}.txt"

        labels = visdrone_to_yolo_labels(
            image_path=image_path,
            annotation_path=annotation_path,
            source_classes=source_classes,
            class_mapping=class_mapping,
        )

        if not labels:
            continue

        unique_name = f"{dataset_dir.name}_{image_path.name}"
        out_image = OUTPUT_ROOT / "images" / split_output_name / unique_name
        out_label = OUTPUT_ROOT / "labels" / split_output_name / f"{dataset_dir.name}_{image_path.stem}.txt"

        copy_or_link(image_path, out_image)
        write_yolo_labels(out_label, labels)
        count += 1

    print(f"[INFO] Wrote {count} labeled samples to {split_output_name}")
    return count


def process_dataset(dataset_dir: Path) -> None:
    cfg = DATASET_CONFIG[dataset_dir.name]

    process_split(
        dataset_dir=dataset_dir,
        split_source_folder=cfg["train_folder"],
        split_output_name="train",
        source_classes=cfg["source_classes"],
        class_mapping=cfg["mapping"],
        annotation_folder=cfg["annotation_folder"],
    )

    process_split(
        dataset_dir=dataset_dir,
        split_source_folder=cfg["val_folder"],
        split_output_name="val",
        source_classes=cfg["source_classes"],
        class_mapping=cfg["mapping"],
        annotation_folder=cfg["annotation_folder"],
    )


def write_dataset_yaml(output_root: Path) -> None:
    yaml_data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": FINAL_CLASS_NAMES,
    }

    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Wrote dataset yaml: {yaml_path}")


def main() -> None:
    prepare_output_dirs(OUTPUT_ROOT)

    for dataset_dir in DATASET_FOLDERS:
        if dataset_dir.exists():
            print(f"\n[INFO] Processing dataset: {dataset_dir.name}")
            process_dataset(dataset_dir)
        else:
            print(f"[WARN] Dataset folder not found: {dataset_dir}")

    write_dataset_yaml(OUTPUT_ROOT)
    print(f"\nDone. Prepared dataset at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()