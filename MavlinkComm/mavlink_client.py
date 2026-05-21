import time
import threading
from typing import Optional, Any

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

		self._connected = False

		# Reader thread control
		self._reader_thread: Optional[threading.Thread] = None
		self._reader_stop = threading.Event()
		self._first_heartbeat = threading.Event()

		# Protects access to cached state
		self._state_lock = threading.Lock()

		# Protects writes/commands sent through the MAVLink connection
		self._send_lock = threading.Lock()

		# Cached latest known vehicle state
		self._state: dict[str, Any] = {
			"armed": None,
			"mode": None,
			"altitude_m": None,
			"lat": None,
			"lon": None,
			"voltage_V": None,
			"battery_remaining": None,
			"gps_fix": None,

			# Timestamps for freshness checks
			"last_heartbeat_time": 0.0,
			"last_position_time": 0.0,
			"last_battery_time": 0.0,
			"last_gps_time": 0.0,
		}

		# Kept for compatibility with your existing diagnostics code
		self._last_heartbeat_time: float = 0.0

		# Optional diagnostics: message counters
		self._message_counts: dict[str, int] = {}

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
				print(
					f"[mavlink] connection attempt {attempt}/{CONNECT_RETRIES} "
					f"-> {self.connection_string}"
				)

				self.master = mavutil.mavlink_connection(self.connection_string)

				self._reader_stop.clear()
				self._first_heartbeat.clear()

				self._reader_thread = threading.Thread(
					target=self._reader_loop,
					daemon=True,
					name="mavlink-reader",
				)
				self._reader_thread.start()

				print("[mavlink] waiting for heartbeat...")

				if not self._first_heartbeat.wait(timeout=HEARTBEAT_TIMEOUT):
					self._stop_reader()
					self.master = None
					raise MAVLinkError(
						f"No heartbeat received within {HEARTBEAT_TIMEOUT}s. "
						"Check cable, baud rate, Pixhawk power, and SERIALx settings."
					)

				self._connected = True

				with self._state_lock:
					last_hb = self._state["last_heartbeat_time"]

				elapsed = time.time() - last_hb
				print(
					f"[mavlink] heartbeat received "
					f"(sysid={self.master.target_system} "
					f"compid={self.master.target_component}, "
					f"{elapsed:.1f}s ago)"
				)
				print("[mavlink] connected successfully.")
				return

			except Exception as e:
				last_exc = e
				print(f"[mavlink] attempt {attempt} failed: {e}")
				self._stop_reader()
				self.master = None
				self._connected = False

				if attempt < CONNECT_RETRIES:
					time.sleep(CONNECT_RETRY_DELAY)

		raise MAVLinkError(
			f"Could not connect after {CONNECT_RETRIES} attempts. "
			f"Last error: {last_exc}"
		)

	def close(self) -> None:
		self._stop_reader()

		if self.master is not None:
			try:
				close_fn = getattr(self.master, "close", None)
				if callable(close_fn):
					close_fn()
			except Exception:
				pass

		self.master = None
		self._connected = False
		print("[mavlink] connection closed.")

	def _stop_reader(self) -> None:
		self._reader_stop.set()

		if self._reader_thread is not None and self._reader_thread.is_alive():
			self._reader_thread.join(timeout=HEARTBEAT_INTERVAL + 1.0)

		self._reader_thread = None

	# ─────────────────────────────────────────
	# SINGLE MAVLINK READER
	# ─────────────────────────────────────────

	def _reader_loop(self) -> None:
		while not self._reader_stop.is_set():
			try:
				if self.master is None:
					time.sleep(0.1)
					continue

				msg = self.master.recv_match(blocking=True, timeout=0.5)

				if msg is None:
					continue

				self._handle_message(msg)

			except Exception as e:
				if not self._reader_stop.is_set():
					print(f"[mavlink] reader error: {e}")
				time.sleep(0.2)

	def _handle_message(self, msg) -> None:
		mtype = msg.get_type()

		if mtype == "BAD_DATA":
			return

		now = time.time()

		with self._state_lock:
			self._message_counts[mtype] = self._message_counts.get(mtype, 0) + 1

			if mtype == "HEARTBEAT":
				self._last_heartbeat_time = now
				self._state["last_heartbeat_time"] = now

				# This is important because we no longer use wait_heartbeat().
				# wait_heartbeat() normally sets target_system/target_component.
				try:
					self.master.target_system = msg.get_srcSystem()
					self.master.target_component = msg.get_srcComponent()
				except Exception:
					pass

				self._state["armed"] = bool(
					msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
				)
				self._state["mode"] = mavutil.mode_string_v10(msg)
				self._first_heartbeat.set()

			elif mtype == "GLOBAL_POSITION_INT":
				self._state["lat"] = msg.lat / 1e7
				self._state["lon"] = msg.lon / 1e7
				self._state["altitude_m"] = msg.relative_alt / 1000.0
				self._state["last_position_time"] = now

			elif mtype == "GPS_RAW_INT":
				self._state["gps_fix"] = msg.fix_type
				self._state["last_gps_time"] = now

			elif mtype == "SYS_STATUS":
				# 65535 means unknown in MAVLink.
				# 0 usually means no battery monitor / not configured in many bench setups.
				if msg.voltage_battery not in (0, 65535):
					self._state["voltage_V"] = msg.voltage_battery / 1000.0
					self._state["last_battery_time"] = now

				if msg.battery_remaining != -1:
					self._state["battery_remaining"] = msg.battery_remaining

			elif mtype == "BATTERY_STATUS":
				# BATTERY_STATUS can provide per-cell voltages.
				# Values are in mV; 65535 means unknown.
				valid_cell_voltages = [
					v for v in msg.voltages
					if v not in (0, 65535)
				]

				if valid_cell_voltages:
					self._state["voltage_V"] = sum(valid_cell_voltages) / 1000.0
					self._state["last_battery_time"] = now

				if msg.battery_remaining != -1:
					self._state["battery_remaining"] = msg.battery_remaining

	# ─────────────────────────────────────────
	# VEHICLE STATE
	# ─────────────────────────────────────────

	def get_vehicle_state(self) -> dict:
		self._require_master()

		with self._state_lock:
			return dict(self._state)

	def get_message_counts(self) -> dict:
		with self._state_lock:
			return dict(self._message_counts)

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

		with self._state_lock:
			last_heartbeat = self._state["last_heartbeat_time"]

		if last_heartbeat <= 0:
			return False

		return (time.time() - last_heartbeat) < HEARTBEAT_INTERVAL * 2

	def get_time_since_last_heartbeat(self) -> Optional[float]:
		with self._state_lock:
			last_heartbeat = self._state["last_heartbeat_time"]

		if last_heartbeat <= 0:
			return None

		return time.time() - last_heartbeat

	# ─────────────────────────────────────────
	# ACTIONS
	# ─────────────────────────────────────────

	def send_go_around(self) -> None:
		master = self._require_master()

		if master is None:
			print("[mavlink] GO_AROUND skipped (disabled).")
			return

		if not self.is_link_alive():
			elapsed = self.get_time_since_last_heartbeat()
			raise MAVLinkError(
				f"Cannot send GO_AROUND: link appears dead "
				f"(last heartbeat {elapsed:.1f}s ago)."
			)

		with self._send_lock:
			master.mav.command_long_send(
				master.target_system,
				master.target_component,
				mavutil.mavlink.MAV_CMD_DO_GO_AROUND,
				0,
				0, 0, 0, 0, 0, 0, 0,
			)

		print("[mavlink] GO_AROUND sent.")

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
		elif action == "GO_AROUND":
			self.send_go_around()
		else:
			raise ValueError(f"Unsupported MAVLink action: {action}")

	def _send_mode_verified(self, mode: str) -> None:
		master = self._require_master()

		if master is None:
			print(f"[mavlink] {mode} skipped (disabled).")
			return

		if not self.is_link_alive():
			elapsed = self.get_time_since_last_heartbeat()
			raise MAVLinkError(
				f"Cannot send {mode}: link appears dead "
				f"(last heartbeat {elapsed:.1f}s ago)."
			)

		for attempt in range(1, MODE_ACK_RETRIES + 1):
			print(f"[mavlink] sending {mode} (attempt {attempt}/{MODE_ACK_RETRIES})")

			with self._send_lock:
				master.set_mode_apm(mode)

			deadline = time.time() + MODE_ACK_TIMEOUT

			while time.time() < deadline:
				current_mode = self.get_current_mode()

				if current_mode == mode:
					print(f"[mavlink] {mode} confirmed by cached heartbeat.")
					return

				time.sleep(0.1)

			print(
				f"[mavlink] {mode} not confirmed within "
				f"{MODE_ACK_TIMEOUT}s, retrying..."
			)

		raise MAVLinkError(
			f"Failed to confirm {mode} after {MODE_ACK_RETRIES} attempts."
		)

	# ─────────────────────────────────────────
	# INTERNAL
	# ─────────────────────────────────────────

	def _require_master(self):
		if not self.enabled:
			return None

		if self.master is None:
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

	# Give the reader thread time to collect state messages
	time.sleep(3.0)

	elapsed = client.get_time_since_last_heartbeat()

	if client.is_link_alive():
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
	fix_labels = {
		0: "No GPS",
		1: "No Fix",
		2: "2D Fix",
		3: "3D Fix",
		4: "DGPS",
		5: "RTK Float",
		6: "RTK Fixed",
	}

	if fix is not None:
		label = fix_labels.get(fix, f"Unknown ({fix})")
		symbol = PASS if fix >= 3 else WARN
		print(f"{symbol} GPS: {label}")
	else:
		print(f"{WARN} Could not read GPS state.")

	voltage = state.get("voltage_V")
	battery_remaining = state.get("battery_remaining")

	if voltage is not None:
		symbol = PASS if voltage >= 10.5 else WARN

		if battery_remaining is not None:
			print(f"{symbol} Battery: {voltage:.2f} V ({battery_remaining}%)")
		else:
			print(f"{symbol} Battery: {voltage:.2f} V")
	else:
		print(
			f"{WARN} Could not read battery voltage. "
			"This is normal if battery monitor is disabled."
		)

	lat = state.get("lat")
	lon = state.get("lon")
	alt = state.get("altitude_m")

	position_valid = (
		lat is not None
		and lon is not None
		and fix is not None
		and fix >= 2
		and not (abs(lat) < 1e-7 and abs(lon) < 1e-7)
	)

	if position_valid:
		alt_text = f"{alt:.1f}m" if alt is not None else "unknown"
		print(f"{INFO} Position: lat={lat:.6f}  lon={lon:.6f}  alt={alt_text}")
	else:
		alt_text = f"{alt:.1f}m" if alt is not None else "unknown"
		print(
			f"{WARN} Global position not valid yet "
			f"(gps_fix={fix}, lat={lat}, lon={lon}, rel_alt={alt_text})."
		)

	print(f"\n{INFO} MAVLink messages seen:")
	counts = client.get_message_counts()

	for name in sorted(counts):
		print(f"  {name}: {counts[name]}")

	print("\n" + "═" * 50)
	print("  Diagnostics complete. Review any [WARN] or [FAIL] above.")
	print("═" * 50 + "\n")

	client.close()


if __name__ == "__main__":
	import sys

	conn = sys.argv[1] if len(sys.argv) > 1 else "/dev/serial0,57600"
	run_diagnostics(conn)