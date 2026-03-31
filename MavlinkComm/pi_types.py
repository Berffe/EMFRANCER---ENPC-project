# types.py
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class FramePacket:
    frame_id:  int
    timestamp: float
    frame_bgr: np.ndarray

@dataclass
class DetectionPacket:
    frame_id:    int
    timestamp:   float
    inference_ms: float
    detections:  list[dict[str, Any]]

@dataclass
class DecisionPacket:
    frame_id:      int
    timestamp:     float
    landing_clear: bool
    reason:        str
    confidence:    float