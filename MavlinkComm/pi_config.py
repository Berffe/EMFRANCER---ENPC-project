from pathlib import Path
import os
from datetime import datetime

MODEL_DIR   = Path("MODEL_best_ncnn")
META_PATH   = MODEL_DIR / "metadata.yaml"
PARAM_PATH  = MODEL_DIR / "model.ncnn.param"
BIN_PATH    = MODEL_DIR / "model.ncnn.bin"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
_mission_ts = datetime.now().strftime("%d_%m_%Y_%H%M%S")
OUTPUT_ROOT = PROJECT_ROOT / f"mission_{_mission_ts}"
VIDEO_DIR   = OUTPUT_ROOT / "video"
LOG_DIR     = OUTPUT_ROOT / "logs"
DEBUG_DIR   = OUTPUT_ROOT / "debug"
LOG_DECISION = True 				# Optional right now

MAIN_SIZE        = (1920, 1080)
LORES_SIZE       = (512, 288)
FRAME_RATE       = 30
SEGMENT_SECONDS  = 120
INFERENCE_PERIOD = 0.5
CONF_THRESH      = 0.25
IOU_THRESH       = 0.45
DISPLAY          = False
SAVE_DEBUG_FRAME_EVERY = 0
CONTAINER_EXT    = "h264"
VIDEO_BITRATE    = 8_000_000
BUFFER_COUNT = 8

MAVLINK_ENABLED = False
MAVLINK_CONNECTION = "serial:/dev/ttyAMA0:57600"
MAVLINK_SUPPRESS_DUPLICATES = True

CONNECT_RETRIES = 5
CONNECT_RETRY_DELAY = 2.0
HEARTBEAT_TIMEOUT = 5.0
HEARTBEAT_INTERVAL = 1.0
MODE_ACK_TIMEOUT = 2.0
MODE_ACK_RETRIES = 3

TELEMETRY_POLL_INTERVAL = 0.5