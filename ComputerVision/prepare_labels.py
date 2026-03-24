from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_FOLDERS = [
	ROOT / "Dataset01",
]

OUTPUT_ROOT = ROOT / "Dataset01_prepared"
COPY_FILES = True  # True = copy, False = symlink

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# FINAL MODEL CLASSES
# ============================================================

FINAL_CLASSES: Dict[str, int] = {
	"person": 0,
	"vehicle": 1,
	"animal": 2,
	"tree": 3,
	"building": 4,
	"landing_zone": 5,
}

FINAL_CLASS_NAMES = {v: k for k, v in FINAL_CLASSES.items()}

# ============================================================
# DATASET 01 MAPPING
# ============================================================

DATASET01_SOURCE_CLASSES: Dict[int, str] = {
	0: "pedestrian",
	1: "people",
	2: "bicycle",
	3: "car",
	4: "van",
	5: "truck",
	6: "tricycle",
	7: "awning-tricycle",
	8: "bus",
	9: "motor",
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

# Fill these later
DATASET02_SOURCE_CLASSES: Dict[int, str] = {
	0: "Vehicle",
	1: "UAP",
	2: "UAI",
	3: "Person",
}

DATASET02_TO_FINAL: Dict[str, str] = {
	"Vehicle": "vehicle",
	"Person": "person",
	"UAP": "landing_zone",   # treat as obstacle
	"UAI": "landing_zone",   # treat as obstacle
}

DATASET03_SOURCE_CLASSES: Dict[int, str] = {}
DATASET03_TO_FINAL: Dict[str, str] = {}

DATASET_CONFIG = {
	"Dataset01": {
		"source_classes": DATASET01_SOURCE_CLASSES,
		"mapping": DATASET01_TO_FINAL,
		"train_folder": "train",
		"val_folder": "valid",
	},
	"Dataset02": {
		"source_classes": DATASET02_SOURCE_CLASSES,
		"mapping": DATASET02_TO_FINAL,
		"train_folder": "train",
		"val_folder": "valid",
	},
	"Dataset03": {
		"source_classes": DATASET03_SOURCE_CLASSES,
		"mapping": DATASET03_TO_FINAL,
		"train_folder": "train",
		"val_folder": "valid",
	},
}

# ============================================================
# HELPERS
# ============================================================

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


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
	rows = []
	if not label_path.exists():
		return rows

	with open(label_path, "r", encoding="utf-8") as f:
		for line_num, line in enumerate(f, start=1):
			parts = line.strip().split()
			if not parts:
				continue
			if len(parts) != 5:
				print(f"[WARN] Invalid label format in {label_path} line {line_num}: {line.strip()}")
				continue

			try:
				cls_id = int(parts[0])
				x = float(parts[1])
				y = float(parts[2])
				w = float(parts[3])
				h = float(parts[4])
				rows.append((cls_id, x, y, w, h))
			except ValueError:
				print(f"[WARN] Could not parse label in {label_path} line {line_num}: {line.strip()}")

	return rows


def remap_labels(
	labels: List[Tuple[int, float, float, float, float]],
	source_classes: Dict[int, str],
	class_mapping: Dict[str, str],
) -> List[Tuple[int, float, float, float, float]]:
	remapped = []

	for cls_id, x, y, w, h in labels:
		source_name = source_classes.get(cls_id)
		if source_name is None:
			continue

		target_name = class_mapping.get(source_name)
		if target_name is None:
			continue

		target_id = FINAL_CLASSES[target_name]
		remapped.append((target_id, x, y, w, h))

	return remapped


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
) -> int:
	images_dir = dataset_dir / split_source_folder / "images"
	labels_dir = dataset_dir / split_source_folder / "labels"

	if not images_dir.exists():
		print(f"[WARN] Missing images directory: {images_dir}")
		return 0

	if not labels_dir.exists():
		print(f"[WARN] Missing labels directory: {labels_dir}")
		return 0

	count = 0
	image_files = find_image_files(images_dir)

	for image_path in image_files:
		label_path = labels_dir / f"{image_path.stem}.txt"
		raw_labels = read_yolo_labels(label_path)
		mapped_labels = remap_labels(raw_labels, source_classes, class_mapping)

		if not mapped_labels:
			continue

		unique_name = f"{dataset_dir.name}_{image_path.name}"
		out_image = OUTPUT_ROOT / "images" / split_output_name / unique_name
		out_label = OUTPUT_ROOT / "labels" / split_output_name / f"{dataset_dir.name}_{image_path.stem}.txt"

		copy_or_link(image_path, out_image)
		write_yolo_labels(out_label, mapped_labels)
		count += 1

	return count


def process_dataset(dataset_dir: Path) -> None:
	dataset_name = dataset_dir.name
	cfg = DATASET_CONFIG.get(dataset_name)

	if cfg is None:
		print(f"[WARN] No config found for {dataset_name}, skipping.")
		return

	source_classes = cfg["source_classes"]
	class_mapping = cfg["mapping"]
	train_folder = cfg["train_folder"]
	val_folder = cfg["val_folder"]

	if not source_classes or not class_mapping:
		print(f"[WARN] Mapping for {dataset_name} is empty, skipping.")
		return

	print(f"\nProcessing {dataset_name}")

	train_count = process_split(
		dataset_dir=dataset_dir,
		split_source_folder=train_folder,
		split_output_name="train",
		source_classes=source_classes,
		class_mapping=class_mapping,
	)

	val_count = process_split(
		dataset_dir=dataset_dir,
		split_source_folder=val_folder,
		split_output_name="val",
		source_classes=source_classes,
		class_mapping=class_mapping,
	)

	print(f"  Converted train samples: {train_count}")
	print(f"  Converted val samples:   {val_count}")


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

	print(f"\nDataset YAML written to: {yaml_path}")


def main() -> None:
	prepare_output_dirs(OUTPUT_ROOT)

	for dataset_dir in DATASET_FOLDERS:
		if dataset_dir.exists():
			process_dataset(dataset_dir)
		else:
			print(f"[WARN] Dataset folder not found: {dataset_dir}")

	write_dataset_yaml(OUTPUT_ROOT)

	print("\nDone.")
	print(f"Prepared dataset available at: {OUTPUT_ROOT}")
	print("Train with:")
	print(OUTPUT_ROOT / "dataset.yaml")


if __name__ == "__main__":
	main()