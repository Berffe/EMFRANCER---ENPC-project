import time

try:
	from pymavlink import mavutil
except ImportError:
	mavutil = None


class MAVLinkClient:
	def __init__(self, connection_string: str, enabled: bool = True):
		self.connection_string = connection_string
		self.enabled = enabled
		self.master = None

	def connect(self) -> None:
		if not self.enabled:
			print("[mavlink] disabled; skipping connection.")
			return

		if mavutil is None:
			raise RuntimeError(
				"pymavlink is not installed. Install it or disable MAVLINK_ENABLED."
			)

		if self.master is not None:
			return

		print(f"[mavlink] connecting -> {self.connection_string}")
		self.master = mavutil.mavlink_connection(self.connection_string)

		try:
			self.master.wait_heartbeat(timeout=5)
			print("[mavlink] heartbeat received.")
		except Exception:
			print("[mavlink] heartbeat not received within timeout; continuing anyway.")

	def close(self) -> None:
		self.master = None

	def _require_master(self):
		if not self.enabled:
			return None
		if self.master is None:
			raise RuntimeError("MAVLink connection is not established.")
		return self.master

	def send_land(self) -> None:
		master = self._require_master()
		if master is None:
			print("[mavlink] LAND skipped (disabled).")
			return

		print("[mavlink] sending LAND")
		master.set_mode_apm("LAND")

	def send_circle(self) -> None:
		master = self._require_master()
		if master is None:
			print("[mavlink] CIRCLE skipped (disabled).")
			return

		print("[mavlink] sending CIRCLE")
		master.set_mode_apm("CIRCLE")

	def send_action(self, action: str) -> None:
		action = action.upper()

		if action == "LAND":
			self.send_land()
		elif action == "CIRCLE":
			self.send_circle()
		else:
			raise ValueError(f"Unsupported MAVLink action: {action}")