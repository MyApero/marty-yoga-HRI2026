import time

import cv2
import mediapipe as mp

from src.live_pose import LivePose


def test_a_result_arrives_without_blocking_and_counts_as_new_once():
    live = LivePose("pose_landmarker_full.task")
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv2.imread("poses/mountain/image.jpg"), cv2.COLOR_BGR2RGB))
    assert live.analyze(image) == (None, False)
    deadline = time.monotonic() + 10
    result, fresh = None, False
    while not fresh and time.monotonic() < deadline:
        time.sleep(0.05)
        result, fresh = live.analyze(image)
    assert fresh and result is not None
    time.sleep(0.01)
    assert live.analyze(image)[0] is not None
