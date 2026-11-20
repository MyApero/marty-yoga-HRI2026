"""Pose detection beside the main loop: the video keeps the camera's pace while the model works on the newest frame."""

import threading
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LivePose:
    def __init__(self, model_path: str):
        self.lock = threading.Lock()
        self.latest = None
        self.fresh = False
        self.last_timestamp_ms = 0
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _on_result(self, result, _image, _timestamp_ms) -> None:
        with self.lock:
            self.latest = result
            self.fresh = True

    def analyze(self, image):
        """(latest result or None, whether it is new since the last call); MediaPipe drops frames while it is busy."""
        self.last_timestamp_ms = max(int(time.monotonic() * 1000), self.last_timestamp_ms + 1)
        self.landmarker.detect_async(image, self.last_timestamp_ms)
        with self.lock:
            fresh, self.fresh = self.fresh, False
            return self.latest, fresh
