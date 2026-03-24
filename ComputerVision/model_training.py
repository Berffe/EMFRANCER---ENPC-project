# model_training.py
"""
Minimal script to train a YOLO model with Ultralytics.

Edit the CONFIG section and run:
	python model_training.py
"""

from ultralytics import YOLO

# ---------------- CONFIG (edit these) ----------------
## Based on which dataset we are using to train and
## which model is the base upon which we improve

DATA_YAML = "Dataset02_prepared/dataset.yaml"   # your dataset config
BASE_MODEL = "runs/detect/runs_set01/terrain-yolov8n/weights/best.pt"         # or "yolo11n.pt", etc.

# EPOCHS -> how many times the model sees the whole training set
# small -> less learning but less overfitting; large -> vice-versa
# first with 20 to check, then 100
EPOCHS = 20			

# IMG_SIZE = n -> resizing YOLO does to feed the images (n x n)
# standard is 640, latter can try bigger (for better performance)
# or smaller (for faster training)
IMG_SIZE = 640

# BATCH_SIZE -> number of images before updating gradient
# if bigger, faster training per epoch but it will need more RAM
# if runs too slow, try 8, 4 or 2
BATCH_SIZE = 4

# "0" for first GPU, "0, 1" for both GPUs, "cpu" for CPU
# can also ommit and let YOLO try cuda
# DEVICE = "0"
PROJECT = "runs_set02"          # output root folder
RUN_NAME = "landing-yolov8n"      # subfolder name inside PROJECT

# -----------------------------------------------------


def main():
	print("Loading base model:", BASE_MODEL)
	model = YOLO(BASE_MODEL)

	print("Starting training...")
	model.train(
		data=DATA_YAML,
		imgsz=IMG_SIZE,
		epochs=EPOCHS,
		batch=BATCH_SIZE,
		project=PROJECT,
		name=RUN_NAME,
	)

	print("\nTraining finished.")
	print("Best weights should be at:")
	print(f"  {PROJECT}/{RUN_NAME}/weights/best.pt")

	# Export the model to NCNN format
	model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'


if __name__ == "__main__":
	main()
