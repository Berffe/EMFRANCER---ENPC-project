"""
DecisionEngine — landing risk assessment for autonomous UAV operations.

Architecture
------------
						┌──────────────────────--┐
DetectionPacket ──►   │  1. Zone classifier    │  bbox → zone (0, 1, 2)
VehicleState   ──►    │  2. Frame risk scorer  │  Σ weight(zone) × confidence
						│  3. Temporal smoother  │  weighted sum over N frames
						│  4. Hard-gate checks   │  altitude / link / confidence
						└──────────┬───────────-─┘
								│
						risk > ABORT_THRESHOLD ?
						├─ YES → action = "GO_AROUND"
						└─ NO  → action = "NONE"  (ArduPilot owns the landing)

Zone geometry (normalized image coordinates, origin top-left)
-------------------------------------------------------------
Zones are defined as trapezoids centred horizontally in the frame.
The trapezoidal shape approximates perspective foreshortening for a
nadir or near-nadir camera. Both zones scale linearly with altitude:
at lower altitude the physical landing footprint fills more of the
frame, so the zones expand proportionally.

Zone 2 — effective landing zone (highest risk weight)
Zone 1 — caution buffer around the landing zone
Zone 0 — rest of the frame (not scored)

		image width
┌──────────────────────┐  ← y = 0
│   zone 0             │
│    ┌────────────┐    │  ← zone 1 top
│    │  ┌──────┐  │    │  ← zone 2 top
│    │  │  Z2  │  │ Z1 │
│    │  └──────┘  │    │  ← zone 2 bottom
│    └────────────┘    │  ← zone 1 bottom
│   zone 0             │
└──────────────────────┘  ← y = 1
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from pi_types import DetectionPacket, DecisionPacket, VehicleStatePacket
from pi_config import (
	WINDOW_SIZE,
	TEMPORAL_DECAY,
	ABORT_THRESHOLD,
	ZONE_WEIGHTS,
	REF_ALTITUDE_M,
	ZONE2_REF,
	ZONE1_REF,
	ALT_SCALE_MIN,
	ALT_SCALE_MAX,
	ALT_SMOOTH_WINDOW,
	HARD_GATE_LINK_DEAD_ABORT,
	VISION_CUTOFF_ALTITUDE_M,
)


# ─────────────────────────────────────────────────────────────
# INTERNAL DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class _FrameRisk:
	"""Risk score computed for a single inference frame."""
	frame_id:  int
	timestamp: float
	raw_risk:  float            # un-normalised frame-level risk score
	max_zone2_conf: float       # highest zone-2 detection confidence this frame
	altitude_m: Optional[float] # smoothed altitude at time of frame


# ─────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────

def _trapezoid_vertices(spec: dict) -> list[tuple[float, float]]:
	"""
	Return the 4 vertices of a symmetric trapezoid in (x, y) normalised coords.
	Vertex order: top-left, top-right, bottom-right, bottom-left (clockwise).
	"""
	cx, cy   = spec["cx"], spec["cy"]
	top_w    = spec["top_w"]
	bot_w    = spec["bot_w"]
	h        = spec["h"]
	half_h   = h / 2.0
	return [
		(cx - top_w / 2,  cy - half_h),   # top-left
		(cx + top_w / 2,  cy - half_h),   # top-right
		(cx + bot_w / 2,  cy + half_h),   # bottom-right
		(cx - bot_w / 2,  cy + half_h),   # bottom-left
	]


def _point_in_polygon(px: float, py: float,
					vertices: list[tuple[float, float]]) -> bool:
	"""
	Ray-casting point-in-polygon test.
	Works for any convex or concave polygon defined by vertices in order.
	"""
	n      = len(vertices)
	inside = False
	x, y   = px, py
	j      = n - 1
	for i in range(n):
		xi, yi = vertices[i]
		xj, yj = vertices[j]
		if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
			inside = not inside
		j = i
	return inside


def _scale_zone(ref: dict, alt_m: float) -> dict:
	"""
	Return a scaled copy of a zone spec for the given altitude.
	Zones expand as altitude decreases (physical footprint fills more of frame).
	"""
	raw_scale = REF_ALTITUDE_M / max(alt_m, 0.5)
	scale     = max(ALT_SCALE_MIN, min(ALT_SCALE_MAX, raw_scale))
	return {
		"cx":    ref["cx"],
		"cy":    ref["cy"],
		"top_w": min(0.95, ref["top_w"] * scale),
		"bot_w": min(0.95, ref["bot_w"] * scale),
		"h":     min(0.90, ref["h"]     * scale),
	}


def _classify_detection(det: dict, alt_m: float) -> int:
	"""
	Return the zone (0, 1, or 2) a detection falls in at the given altitude.
	Uses the centre of the bounding box as the representative point.
	"""
	x1, y1, x2, y2 = det["bbox"]
	# Normalise bbox centre to [0, 1] using the lores frame dimensions
	# Note: bbox coords are in lores pixel space (e.g. 512×288)
	# We normalise here; caller must ensure bbox is in pixel coords.
	frame_w, frame_h = 512.0, 288.0   # matches LORES_SIZE in pi_config
	cx = ((x1 + x2) / 2.0) / frame_w
	cy = ((y1 + y2) / 2.0) / frame_h

	# Check innermost zone first
	z2_spec  = _scale_zone(ZONE2_REF, alt_m)
	z2_verts = _trapezoid_vertices(z2_spec)
	if _point_in_polygon(cx, cy, z2_verts):
		return 2

	z1_spec  = _scale_zone(ZONE1_REF, alt_m)
	z1_verts = _trapezoid_vertices(z1_spec)
	if _point_in_polygon(cx, cy, z1_verts):
		return 1

	return 0


# ─────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────

class DecisionEngine:
	"""
	Stateful landing risk engine.

	Call update() on every DetectionPacket received during a LAND sequence.
	The engine maintains an internal rolling window and returns a
	DecisionPacket whose action is either "NONE" (continue landing) or
	"GO_AROUND" (abort — Pi will issue MAV_CMD_DO_GO_AROUND).
	"""

	def __init__(self):
		self._history: deque[_FrameRisk] = deque(maxlen=WINDOW_SIZE)
		self._alt_buffer: deque[float]   = deque(maxlen=ALT_SMOOTH_WINDOW)

	# ── public ────────────────────────────────────────────────

	def update(
		self,
		pkt: DetectionPacket,
		vehicle_state: Optional[VehicleStatePacket] = None,
	) -> DecisionPacket:
		alt_raw = vehicle_state.altitude_m if vehicle_state else None
		alt_smooth = self._update_altitude(alt_raw)
		alt_for_scoring = alt_smooth if alt_smooth is not None else REF_ALTITUDE_M

		# If we are too close to the ground, stop using vision-based abort logic
		# and let ArduPilot complete the landing natively.
		if alt_smooth is not None and alt_smooth <= VISION_CUTOFF_ALTITUDE_M:
			return DecisionPacket(
				frame_id=pkt.frame_id,
				timestamp=pkt.timestamp,
				landing_clear=True,
				action="NONE",
				reason=(
					f"vision_disabled_below_cutoff: "
					f"alt={alt_smooth:.2f}m <= {VISION_CUTOFF_ALTITUDE_M:.2f}m"
				),
				confidence=1.0,
			)

		# ── 1. Score this frame ────────────────────────────────
		frame_risk, max_z2_conf = self._score_frame(pkt.detections, alt_for_scoring)

		self._history.append(_FrameRisk(
			frame_id    = pkt.frame_id,
			timestamp   = pkt.timestamp,
			raw_risk    = frame_risk,
			max_zone2_conf = max_z2_conf,
			altitude_m  = alt_smooth,
		))

		# ── 2. Temporal weighted score ─────────────────────────
		temporal_score = self._temporal_score()

		# ── 3. Hard abort gates ────────────────────────────────
		hard_abort, hard_reason = self._check_hard_gates(
			alt_smooth, max_z2_conf, vehicle_state
		)

		# ── 4. Final decision ──────────────────────────────────
		if hard_abort:
			abort    = True
			reason   = hard_reason
			confidence = 1.0
		elif temporal_score > ABORT_THRESHOLD:
			abort    = True
			reason   = f"risk_score={temporal_score:.3f} > threshold={ABORT_THRESHOLD}"
			confidence = min(1.0, temporal_score)
		else:
			abort    = False
			reason   = f"risk_score={temporal_score:.3f} <= threshold={ABORT_THRESHOLD}"
			confidence = 1.0 - temporal_score

		action = "GO_AROUND" if abort else "NONE"

		return DecisionPacket(
			frame_id      = pkt.frame_id,
			timestamp     = pkt.timestamp,
			landing_clear = not abort,
			action        = action,
			reason        = reason,
			confidence    = round(confidence, 4),
		)

	def reset(self) -> None:
		"""Call between landing attempts to clear history."""
		self._history.clear()
		self._alt_buffer.clear()

	# ── private ───────────────────────────────────────────────

	def _update_altitude(self, alt_m: Optional[float]) -> Optional[float]:
		"""Push new altitude reading and return smoothed value."""
		if alt_m is not None and alt_m >= 0.0:
			self._alt_buffer.append(alt_m)
		if not self._alt_buffer:
			return None
		return sum(self._alt_buffer) / len(self._alt_buffer)

	def _score_frame(
		self,
		detections: list[dict],
		alt_m: float,
	) -> tuple[float, float]:
		"""
		Compute a single-frame risk score as a weighted sum over detections.

		Each detection contributes:  ZONE_WEIGHTS[zone] × detection.confidence

		Returns (frame_risk, max_zone2_confidence).
		"""
		frame_risk   = 0.0
		max_z2_conf  = 0.0

		for det in detections:
			zone   = _classify_detection(det, alt_m)
			weight = ZONE_WEIGHTS[zone]
			conf   = float(det["score"])
			frame_risk  += weight * conf
			if zone == 2:
				max_z2_conf = max(max_z2_conf, conf)

		return frame_risk, max_z2_conf

	def _temporal_score(self) -> float:
		"""
		Exponentially-weighted average of frame risk scores.

		Newest frame has weight 1.0; each older frame is multiplied
		by TEMPORAL_DECAY, so the oldest frame has weight ≈ TEMPORAL_DECAY^(N-1).
		The result is normalised to [0, 1] by dividing by the sum of weights.
		"""
		if not self._history:
			return 0.0

		frames  = list(self._history)   # oldest → newest
		n       = len(frames)
		weights = [TEMPORAL_DECAY ** (n - 1 - i) for i in range(n)]

		total_w = sum(weights)
		if total_w == 0.0:
			return 0.0

		weighted_sum = sum(w * f.raw_risk for w, f in zip(weights, frames))
		# Normalise by the max possible score (all zone-2 detections, conf=1)
		# so the threshold lives in [0,1] regardless of ZONE_WEIGHTS scale.
		max_possible = ZONE_WEIGHTS[2]
		if max_possible == 0.0:
			return 0.0

		raw_avg = weighted_sum / total_w
		return min(1.0, raw_avg / max_possible)

	def _check_hard_gates(
		self,
		alt_smooth: Optional[float],
		max_z2_conf: float,
		vehicle_state: Optional[VehicleStatePacket],
	) -> tuple[bool, str]:
		"""
		Hard-coded abort conditions that bypass the risk score entirely.
		Returns (should_abort, reason_string).
		"""
		# Gate 1: MAVLink link is dead — we cannot trust state, abort
		if HARD_GATE_LINK_DEAD_ABORT:
			link_alive = vehicle_state.link_alive if vehicle_state else False
			if not link_alive:
				return True, "hard_gate: mavlink_link_dead"

		# # Gate 2: Very close to ground AND a confident zone-2 detection
		# if alt_smooth is not None and alt_smooth <= HARD_GATE_MIN_ALTITUDE_M:
		# 	if max_z2_conf >= HARD_GATE_ZONE2_CONF_THRESH:
		# 		return True, (
		# 			f"hard_gate: zone2_conf={max_z2_conf:.2f} "
		# 			f">= {HARD_GATE_ZONE2_CONF_THRESH} "
		# 			f"at alt={alt_smooth:.1f}m"
		# 		)

		# Gate 3: Altitude unavailable for too long — conservative abort
		if alt_smooth is None and len(self._history) >= WINDOW_SIZE:
			return True, "hard_gate: altitude_unavailable_full_window"

		return False, ""