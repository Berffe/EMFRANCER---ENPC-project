# config.py
from pathlib import Path

MODEL_DIR   = Path("MODEL_best_ncnn")
META_PATH   = MODEL_DIR / "metadata.yaml"
PARAM_PATH  = MODEL_DIR / "model.ncnn.param"
BIN_PATH    = MODEL_DIR / "model.ncnn.bin"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_ROOT = PROJECT_ROOT / "01_04_2026_mission001"
VIDEO_DIR   = OUTPUT_ROOT / "video"
LOG_DIR     = OUTPUT_ROOT / "logs"
DEBUG_DIR   = OUTPUT_ROOT / "debug"

MAIN_SIZE        = (1920, 1080)
LORES_SIZE       = (512, 288)
FRAME_RATE       = 30
SEGMENT_SECONDS  = 300
INFERENCE_PERIOD = 0.5
CONF_THRESH      = 0.25
IOU_THRESH       = 0.45
DISPLAY          = False
SAVE_DEBUG_FRAME_EVERY = 0
CONTAINER_EXT    = "h264"
VIDEO_BITRATE    = 8_000_000
BUFFER_COUNT = 8