from ultralytics import YOLO
from pathlib import Path

PROJECT =  Path("runs\\detect\\runs_set01")          # output root folder
RUN_NAME =  Path("terrain-yolov8n")      # subfolder name inside PROJECT
BEST_WEIGHT = PROJECT/RUN_NAME/"weights/best.pt"
print(BEST_WEIGHT)

# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")
model = YOLO(BEST_WEIGHT)

# Export the model to NCNN format
model.export(format="ncnn", imgsz=320)  # creates 'yolov8n_ncnn_model'