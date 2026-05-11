from pathlib import Path
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# SAVING LOCATION
# ─────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────
# IMAGE/ INFERENCE QUALITY PARAMETERS
# ─────────────────────────────────────────────────────────────


MAIN_SIZE        = (1920, 1080)
LORES_SIZE       = (512, 288) ## Try (320, 192)
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
INF_THREADS = 2


# ─────────────────────────────────────────────────────────────
# MAVLINK COMMUNICATION PARAMETERS
# ─────────────────────────────────────────────────────────────

MAVLINK_ENABLED = False
MAVLINK_CONNECTION = "/dev/ttyAMA0,57600"
MAVLINK_SUPPRESS_DUPLICATES = True

CONNECT_RETRIES = 5
CONNECT_RETRY_DELAY = 2.0
HEARTBEAT_TIMEOUT = 5.0
HEARTBEAT_INTERVAL = 1.0
MODE_ACK_TIMEOUT = 2.0
MODE_ACK_RETRIES = 3

TELEMETRY_POLL_INTERVAL = 0.5


# ─────────────────────────────────────────────────────────────
# TUNABLE DECISION PARAMETERS
# ─────────────────────────────────────────────────────────────

# --- Temporal window ---
WINDOW_SIZE = 10            # number of inference frames kept in history
TEMPORAL_DECAY = 0.75       # exponential decay factor; newest frame weight = 1.0,
							# oldest ≈ TEMPORAL_DECAY^(WINDOW_SIZE-1)

# --- Abort threshold ---
ABORT_THRESHOLD = 0.30      # normalised [0, 1]; tune after ground tests

# --- Zone risk weights ---
ZONE_WEIGHTS = {
	0: 0.00,   # outside both zones — not scored
	1: 0.35,   # caution zone — partial weight
	2: 1.00,   # landing zone — full weight
}

# --- Zone geometry at reference altitude (normalised image coords) ---
# Trapezoid spec: cx (horizontal centre), cy (vertical centre),
#                 top_w (top edge width), bot_w (bottom edge width), h (height)
# Both zones are centred at the same cx/cy.
REF_ALTITUDE_M  = 10.0      # metres — geometry defined at this altitude
ZONE2_REF = dict(cx=0.50, cy=0.58, top_w=0.18, bot_w=0.28, h=0.28)
ZONE1_REF = dict(cx=0.50, cy=0.58, top_w=0.34, bot_w=0.50, h=0.46)

# Altitude scaling bounds (prevents runaway expansion / invisible zones)
ALT_SCALE_MIN = 0.25        # at very high altitude, zones don't vanish entirely
ALT_SCALE_MAX = 2.50        # at very low altitude, zones don't overflow frame

# --- Altitude smoothing ---
ALT_SMOOTH_WINDOW = 6       # frames over which to average altitude readings

# --- Hard abort gates ---
# If ANY of these trigger, we abort regardless of the risk score.
# HARD_GATE_MIN_ALTITUDE_M      = 2.0    # below this, any zone-2 detection → abort
# HARD_GATE_ZONE2_CONF_THRESH   = 0.80   # single very-confident zone-2 hit → abort
HARD_GATE_LINK_DEAD_ABORT     = True   # abort if MAVLink link is reported dead
VISION_CUTOFF_ALTITUDE_M = 3.0