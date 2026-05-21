from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class FramePacket:
	frame_id: int
	timestamp: float
	segment_index: int
	segment_t_sec: float
	estimated_main_frame_idx: int
	frame_bgr: np.ndarray


@dataclass
class DetectionPacket:
	frame_id: int
	timestamp: float
	segment_index: int
	segment_t_sec: float
	estimated_main_frame_idx: int
	inference_ms: float
	detections: list[dict[str, Any]]


@dataclass
class DecisionPacket:
	frame_id: int
	timestamp: float
	landing_clear: bool
	action: str
	reason: str
	confidence: float


@dataclass
class CommandPacket:
	frame_id: int
	timestamp: float
	action: str
	reason: str
	confidence: float


@dataclass
class VehicleStatePacket:
	timestamp: float
	link_alive: bool
	armed: bool | None
	mode: str | None
	altitude_m: float | None
	lat: float | None
	lon: float | None
	voltage_V: float | None
	gps_fix: int | None
	battery_remaining: int | None = None