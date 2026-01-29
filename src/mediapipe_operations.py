import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


def setup_landmarker(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)


def apply_film_effect(frame, settings):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    film = cv2.convertScaleAbs(
        gray, alpha=settings["contrast"], beta=settings["brightness"]
    )
    film = cv2.cvtColor(film, cv2.COLOR_GRAY2BGR)
    noise = np.random.randint(0, settings["grain_intensity"], film.shape, dtype="uint8")
    return cv2.add(film, noise)
