import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from dataclasses import dataclass

from src.utils import load_toml
from src.video_feedback import draw_skeleton

@dataclass
class Pose:
    name: str
    mediapipe_pose: str
    be_careful_at: str

class HeadMaster:
    def __init__(self):
        self.camera = self.init_camera()
        self.marty = self.init_marty()
        self.pose = None
        self.pose_duration = 10  # seconds
        self.pose_correct_timer = 0

    def init_camera(self):
        pass

    def init_marty(self):
        pass

    def capture_image_from_camera(self):
        pass

    def process_image(self, image):
        pass

    def load_pose(self, pose):
        self.pose = pose

    def do_pose(self):
        while self.pose_correct_timer < self.pose_duration:
            camera_image = self.capture_image_from_camera()
            processed_image = self.process_image(camera_image)


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


def main():
    # Initial Setup
    CONFIG_FILE = "config.toml"
    POSES_FOLDER = "poses/"

    config = load_toml(CONFIG_FILE)
    poses_list = ["warrior", "chair"]
    poses = {
        pose_name: load_toml(os.path.join(POSES_FOLDER, pose_name, "pose.toml"))["pose"]
        for pose_name in poses_list
    }
    last_config_time = os.path.getmtime(CONFIG_FILE)
    cap = cv2.VideoCapture(2)
    landmarker = setup_landmarker(config["model_path"])

    pose_name = "warrior"  # Currently hardcoded; can be extended to switch poses

    while True:
        # Hot Reload logic
        curr_time = os.path.getmtime(CONFIG_FILE)
        if curr_time > last_config_time:
            try:
                config = load_toml()
                last_config_time = curr_time
                print("Skeleton updated from TOML!")
            except:
                pass

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        output_frame = apply_film_effect(frame, config["film_settings"])
        if result.pose_landmarks:
            draw_skeleton(output_frame, result.pose_landmarks, config, poses[pose_name])

        big_frame = cv2.resize(
            output_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        cv2.imshow("Skeleton", big_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# say_bonjour()

# for pose in poses:
#     print(f"Now demonstrating the {pose.name} pose.")
#     headmaster.load_pose(pose)
#     headmaster.do_pose()
