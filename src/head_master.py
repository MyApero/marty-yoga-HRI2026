import cv2
import mediapipe as mp
import os
import time
from src.utils import load_toml
from src.video_feedback import draw_skeleton
from src.mediapipe_operations import setup_landmarker, apply_film_effect
from src.marty import Marty
import logging

# Initial Setup
CONFIG_FILE = "config.toml"
POSES_FOLDER = "poses/"
POSES_LIST = ["warrior", "chair"]


class HeadMaster:
    def __init__(self, camera_index, pose_duration=10, logging_level=logging.INFO):
        self.camera = self.init_camera(camera_index)
        self.marty = self.init_marty()
        self.pose = None
        self.pose_duration = pose_duration  # seconds
        self.pose_correct_timer = 0
        self.config = load_toml(CONFIG_FILE)
        self.poses = {
            pose_name: load_toml(os.path.join(POSES_FOLDER, pose_name, "pose.toml"))[
                "pose"
            ]
            for pose_name in POSES_LIST
        }

        self.landmarker = setup_landmarker(self.config["model_path"])

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging_level)

    def init_camera(self, camera_index=0):
        return cv2.VideoCapture(camera_index)

    def init_marty(self):
        return Marty()

    def capture_image_from_camera(self):
        success, frame = self.camera.read()
        if not success:
            raise RuntimeError("Failed to read from camera.")

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return mp_image

    def filter_image(self, image):
        return apply_film_effect(image.numpy_view(), self.config["film_settings"])

    def process_image(self, show_landmarks=False, timer_text=""):
        camera_image = self.capture_image_from_camera()
        if show_landmarks:
            result = self.analyze_image(camera_image)
        output_frame = self.filter_image(camera_image)
        if show_landmarks and result.pose_landmarks:
            draw_skeleton(
                output_frame,
                result.pose_landmarks,
                self.config,
                self.poses[self.pose],
            )
        if show_landmarks:
            cv2.putText(
                output_frame,
                timer_text,
                (output_frame.shape[1] - 200, output_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        frame = cv2.resize(
            output_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        cv2.imshow("Video Feedback", frame)

    def analyze_image(self, image):
        timestamp = int(time.time() * 1000)
        return self.landmarker.detect_for_video(image, timestamp)

    def load_pose(self, pose):
        self.pose = pose

    def do_pose(self):
        start_time = time.time()
        timer_text = ""
        while self.pose_correct_timer < self.pose_duration:
            elapsed_time = time.time() - start_time
            timer_text = f"Time in pose: {int(elapsed_time)}s"
            self.process_image(show_landmarks=True, timer_text=timer_text)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()
