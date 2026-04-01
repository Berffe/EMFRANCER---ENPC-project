from typing import Optional
from pi_types import DetectionPacket, DecisionPacket


class DecisionEngine:
	def __init__(self, obstacle_classes: Optional[set[int]] = None):
		self.obstacle_classes = obstacle_classes or set()

	def update(self, pkt: DetectionPacket) -> DecisionPacket:
		if not pkt.detections:
			return DecisionPacket(
				frame_id=pkt.frame_id,
				timestamp=pkt.timestamp,
				landing_clear=True,
				action="LAND",
				reason="no_detections",
				confidence=0.8,
			)

		return DecisionPacket(
			frame_id=pkt.frame_id,
			timestamp=pkt.timestamp,
			landing_clear=False,
			action="CIRCLE",
			reason="detections_present",
			confidence=0.9,
		)