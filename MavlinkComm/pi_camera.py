# camera.py
import threading
from pathlib import Path
import cv2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput, FileOutput
from pi_config import (MAIN_SIZE, LORES_SIZE, FRAME_RATE,
					VIDEO_BITRATE, CONTAINER_EXT, VIDEO_DIR, BUFFER_COUNT)

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
		self._segment_index  = 0
		self._recording_lock = threading.Lock()

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

	def capture_lores_bgr(self):
		lores = self.picam2.capture_array("lores")
		return cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)

	def start_new_segment(self) -> Path:
		with self._recording_lock:
			try:
				self.picam2.stop_recording()
			except Exception:
				pass
			out_path = VIDEO_DIR / f"segment_{self._segment_index:04d}.{CONTAINER_EXT}"
			self.picam2.start_recording(self.encoder, FileOutput(str(out_path)))
			# self.picam2.start_recording(self.encoder, FfmpegOutput(str(out_path)))
			self._segment_index += 1
			return out_path