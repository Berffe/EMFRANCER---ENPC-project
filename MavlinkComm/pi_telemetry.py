import threading
from typing import Optional

from pi_types import VehicleStatePacket


class TelemetryState:
	def __init__(self):
		self._lock = threading.Lock()
		self._latest: Optional[VehicleStatePacket] = None

	def update(self, pkt: VehicleStatePacket) -> None:
		with self._lock:
			self._latest = pkt

	def get_snapshot(self) -> Optional[VehicleStatePacket]:
		with self._lock:
			return self._latest