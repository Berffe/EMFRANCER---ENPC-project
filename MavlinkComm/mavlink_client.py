import time
import threading
from typing import Optional
from pi_config import (
	CONNECT_RETRIES,
	CONNECT_RETRY_DELAY,
	HEARTBEAT_TIMEOUT,
	HEARTBEAT_INTERVAL,
	MODE_ACK_TIMEOUT,
	MODE_ACK_RETRIES,
)

try:
	from pymavlink import mavutil
except ImportError:
	mavutil = None


class MAVLinkError(Exception):
	"""Raised for unrecoverable MAVLink errors so callers can react specifically."""
	pass


class MAVLinkClient:
	def __init__(self, connection_string: str, enabled: bool = True):
		self.connection_string = connection_string
		self.enabled = enabled
		self.master: Optional[object] = None

		self._last_heartbeat_time: float = 0.0
		self._watchdog_thread: Optional[threading.Thread] = None
		self._watchdog_stop = threading.Event()
		self._connected = False

		# Serializes recv_match()/set_mode_apm() access across watchdog,
		# telemetry reads, and command verification.
		self._io_lock = threading.Lock()

	# ─────────────────────────────────────────
	# CONNECTION
	# ─────────────────────────────────────────

	def connect(self) -> None:
		if not self.enabled:
			print("[mavlink] disabled; skipping connection.")
			return

		if mavutil is None:
			raise MAVLinkError(
				"pymavlink is not installed. "
				"Run: pip install pymavlink --break-system-packages"
			)

		if self._connected:
			return

		last_exc = None
		for attempt in range(1, CONNECT_RETRIES + 1):
			try:
				print(f"[mavlink] connection attempt {attempt}/{CONNECT_RETRIES} "
					f"-> {self.connection_string}")
				self.master = mavutil.mavlink_connection(self.connection_string)
				self._wait_for_heartbeat()
				self._connected = True
				self._start_watchdog()
				print("[mavlink] connected successfully.")
				return
			except Exception as e:
				last_exc = e
				print(f"[mavlink] attempt {attempt} failed: {e}")
				self.master = None
				if attempt < CONNECT_RETRIES:
					time.sleep(CONNECT_RETRY_DELAY)

		raise MAVLinkError(
			f"Could not connect after {CONNECT_RETRIES} attempts. "
			f"Last error: {last_exc}"
		)

	def close(self) -> None:
		self._watchdog_stop.set()
		if self._watchdog_thread is not None:
			self._watchdog_thread.join(timeout=HEARTBEAT_INTERVAL + 1.0)
		self.master = None
		self._connected = False
		print("[mavlink] connection closed.")

	# ─────────────────────────────────────────
	# HEARTBEAT HANDLING
	# ─────────────────────────────────────────

	def _wait_for_heartbeat(self) -> None:
		print("[mavlink] waiting for heartbeat...")
		hb = self.master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT)
		if hb is None:
			raise MAVLinkError(
				f"No heartbeat received within {HEARTBEAT_TIMEOUT}s. "
				"Check cable, baud rate, and Pixhawk power."
			)
		self._last_heartbeat_time = time.time()
		print(f"[mavlink] heartbeat received "
			f"(sysid={self.master.target_system} "
			f"compid={self.master.target_component})")

	def _watchdog_loop(self) -> None:
		while not self._watchdog_stop.wait(HEARTBEAT_INTERVAL):
			if not self._connected or self.master is None:
				break

			try:
				with self._io_lock:
					msg = self.master.recv_match(type="HEARTBEAT", blocking=False)
				if msg is not None:
					self._last_heartbeat_time = time.time()
			except Exception as e:
				print(f"[mavlink] watchdog read error: {e}")

			elapsed = time.time() - self._last_heartbeat_time
			if elapsed > HEARTBEAT_INTERVAL * 2:
				print(f"[mavlink] WARNING: no heartbeat for {elapsed:.1f}s — "
					"link may be lost.")

	def _start_watchdog(self) -> None:
		self._watchdog_stop.clear()
		self._watchdog_thread = threading.Thread(
			target=self._watchdog_loop,
			daemon=True,
			name="mavlink-watchdog",
		)
		self._watchdog_thread.start()

	# ─────────────────────────────────────────
	# VEHICLE STATE
	# ─────────────────────────────────────────

	def get_vehicle_state(self) -> dict:
		master = self._require_master()
		if master is None:
			return {}

		state = {
			"armed": None,
			"mode": None,
			"altitude_m": None,
			"lat": None,
			"lon": None,
			"voltage_V": None,
			"gps_fix": None,
		}

		with self._io_lock:
			hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=2.0)
			if hb:
				self._last_heartbeat_time = time.time()
				state["armed"] = bool(
					hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
				)
				state["mode"] = mavutil.mode_string_v10(hb)

			pos = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0)
			if pos:
				state["lat"] = pos.lat / 1e7
				state["lon"] = pos.lon / 1e7
				state["altitude_m"] = pos.relative_alt / 1000.0

			sys = master.recv_match(type="SYS_STATUS", blocking=True, timeout=2.0)
			if sys:
				state["voltage_V"] = sys.voltage_battery / 1000.0

			gps = master.recv_match(type="GPS_RAW_INT", blocking=True, timeout=2.0)
			if gps:
				state["gps_fix"] = gps.fix_type

		return state

	def get_current_mode(self) -> Optional[str]:
		return self.get_vehicle_state().get("mode")

	def get_position(self) -> tuple[Optional[float], Optional[float]]:
		state = self.get_vehicle_state()
		return state.get("lat"), state.get("lon")

	def get_altitude_m(self) -> Optional[float]:
		return self.get_vehicle_state().get("altitude_m")

	def is_link_alive(self) -> bool:
		if not self._connected or self.master is None:
			return False
		return (time.time() - self._last_heartbeat_time) < HEARTBEAT_INTERVAL * 2

	# ─────────────────────────────────────────
	# ACTIONS
	# ─────────────────────────────────────────

	def send_land(self) -> None:
		self._send_mode_verified("LAND")

	def send_circle(self) -> None:
		self._send_mode_verified("CIRCLE")

	def send_action(self, action: str) -> None:
		action = action.upper()
		if action == "LAND":
			self.send_land()
		elif action == "CIRCLE":
			self.send_circle()
		else:
			raise ValueError(f"Unsupported MAVLink action: {action}")

	def _send_mode_verified(self, mode: str) -> None:
		master = self._require_master()
		if master is None:
			print(f"[mavlink] {mode} skipped (disabled).")
			return

		if not self.is_link_alive():
			raise MAVLinkError(
				f"Cannot send {mode}: link appears dead "
				f"(last heartbeat {time.time() - self._last_heartbeat_time:.1f}s ago)."
			)

		for attempt in range(1, MODE_ACK_RETRIES + 1):
			print(f"[mavlink] sending {mode} (attempt {attempt}/{MODE_ACK_RETRIES})")

			with self._io_lock:
				master.set_mode_apm(mode)

				deadline = time.time() + MODE_ACK_TIMEOUT
				while time.time() < deadline:
					hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
					if hb is None:
						continue
					self._last_heartbeat_time = time.time()
					current_mode = mavutil.mode_string_v10(hb)
					if current_mode == mode:
						print(f"[mavlink] {mode} confirmed by heartbeat.")
						return

			print(f"[mavlink] {mode} not confirmed within {MODE_ACK_TIMEOUT}s, retrying...")

		raise MAVLinkError(
			f"Failed to confirm {mode} after {MODE_ACK_RETRIES} attempts."
		)

	# ─────────────────────────────────────────
	# INTERNAL
	# ─────────────────────────────────────────

	def _require_master(self):
		if not self.enabled:
			return None
		if self.master is None or not self._connected:
			raise MAVLinkError("MAVLink connection is not established. Call connect() first.")
		return self.master


def run_diagnostics(connection_string: str) -> None:
	PASS = "\033[92m[PASS]\033[0m"
	FAIL = "\033[91m[FAIL]\033[0m"
	WARN = "\033[93m[WARN]\033[0m"
	INFO = "\033[94m[INFO]\033[0m"

	print("\n" + "═" * 50)
	print("  MAVLink Diagnostics")
	print("═" * 50)

	client = MAVLinkClient(connection_string, enabled=True)

	print(f"\n{INFO} Testing connection to {connection_string}...")
	try:
		client.connect()
		print(f"{PASS} Connection established.")
	except MAVLinkError as e:
		print(f"{FAIL} Connection failed: {e}")
		return

	time.sleep(1.0)
	if client.is_link_alive():
		elapsed = time.time() - client._last_heartbeat_time
		print(f"{PASS} Link alive (last heartbeat {elapsed:.1f}s ago).")
	else:
		print(f"{FAIL} Link appears dead after connection.")

	print(f"\n{INFO} Reading vehicle state...")
	state = client.get_vehicle_state()

	if state.get("mode") is not None:
		print(f"{PASS} Flight mode: {state['mode']}")
	else:
		print(f"{WARN} Could not read flight mode.")

	armed = state.get("armed")
	if armed is not None:
		status = "ARMED" if armed else "DISARMED"
		symbol = WARN if armed else PASS
		print(f"{symbol} Vehicle is {status}.")
	else:
		print(f"{WARN} Could not read arm status.")

	fix = state.get("gps_fix")
	fix_labels = {0: "No GPS", 1: "No Fix", 2: "2D Fix",
				3: "3D Fix", 4: "DGPS", 5: "RTK Float", 6: "RTK Fixed"}
	if fix is not None:
		label = fix_labels.get(fix, f"Unknown ({fix})")
		symbol = PASS if fix >= 3 else WARN
		print(f"{symbol} GPS: {label}")
	else:
		print(f"{WARN} Could not read GPS state.")

	voltage = state.get("voltage_V")
	if voltage is not None:
		symbol = PASS if voltage >= 10.5 else WARN
		print(f"{symbol} Battery: {voltage:.2f} V")
	else:
		print(f"{WARN} Could not read battery voltage.")

	lat, lon, alt = state.get("lat"), state.get("lon"), state.get("altitude_m")
	if lat is not None and lon is not None:
		print(f"{INFO} Position: lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f}m")
	else:
		print(f"{WARN} Could not read position.")

	print("\n" + "═" * 50)
	print("  Diagnostics complete. Review any [WARN] or [FAIL] above.")
	print("═" * 50 + "\n")

	client.close()


if __name__ == "__main__":
	import sys

	conn = sys.argv[1] if len(sys.argv) > 1 else "serial:/dev/serial0:57600"
	run_diagnostics(conn)