import threading
import time
from pathlib import Path

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import PyavOutput

from pi_config import (
	MAIN_SIZE,
	LORES_SIZE,
	FRAME_RATE,
	VIDEO_BITRATE,
	CONTAINER_EXT,
	VIDEO_DIR,
	BUFFER_COUNT,
)


class CameraManager:
	def __init__(self):
		self.picam2 = Picamera2()
		config = self.picam2.create_video_configuration(
			main={"size": MAIN_SIZE},
			lores={"size": LORES_SIZE, "format": "YUV420"},
			controls={"FrameRate": FRAME_RATE},
			buffer_count=BUFFER_COUNT,
		)
		self.picam2.configure(config)
		self.encoder = H264Encoder(bitrate=VIDEO_BITRATE, repeat=True, iperiod=15)

		self._next_segment_index = 0
		self._current_segment_index = -1
		self._current_segment_path = None
		self._current_segment_start_ts = 0.0

		self._recording_lock = threading.Lock()
		self._segment_lock = threading.Lock()

		# Cache raw lores frame, not converted BGR
		self._latest_lores_raw = None
		self._latest_lores_lock = threading.Lock()

		self.picam2.post_callback = self._post_callback

	def _post_callback(self, request):
		try:
			lores = request.make_array("lores")
			with self._latest_lores_lock:
				self._latest_lores_raw = lores
		except Exception:
			pass

	def start(self) -> None:
		self.picam2.start()
		self.start_new_segment()

	def stop(self) -> None:
		with self._recording_lock:
			try:
				self.picam2.stop_recording()
			except Exception:
				pass
		self.picam2.stop()

	def get_latest_lores_raw(self):
		with self._latest_lores_lock:
			if self._latest_lores_raw is None:
				return None
			return self._latest_lores_raw.copy()

	def get_current_segment_info(self):
		with self._segment_lock:
			return (
				self._current_segment_index,
				self._current_segment_start_ts,
				self._current_segment_path,
			)

	def start_new_segment(self) -> Path:
		with self._recording_lock:
			try:
				self.picam2.stop_recording()
			except Exception:
				pass

			segment_index = self._next_segment_index
			out_path = VIDEO_DIR / f"segment_{segment_index:04d}.{CONTAINER_EXT}"
			self.picam2.start_recording(
				self.encoder,
				PyavOutput(str(out_path), format="matroska"),
			)

			start_ts = time.time()
			with self._segment_lock:
				self._current_segment_index = segment_index
				self._current_segment_path = out_path
				self._current_segment_start_ts = start_ts

			self._next_segment_index += 1
			return out_path