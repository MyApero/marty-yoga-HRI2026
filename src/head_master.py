import cv2
import os
import time
import logging
import sys
import toml

from src.utils import load_toml
from src.video_feedback import draw_skeleton
from src.mediapipe_operations import setup_landmarker, apply_film_effect
from src.feedback_preprocess import get_feedbacks_from_run
from src.marty import MyMarty
from src.speak import Speak
from src.camera import capture_image_from_camera
from src.utils import get_color_gradient

# Initial Setup
CONFIG_FILE = "config.toml"
POSES_FOLDER = "poses/"
POSES_LIST = ["right_warrior2", "left_warrior2", "chair", "mountain"]

MARGIN_BEFORE_CORRECTION_FEEDBACK_S = 5
TIME_GENERATION_END_FEEDBACK_S = 12.0
POSE_DURATION_S = 50
SEND_CORRECTION_THRESHOLD = (
    0.6  # the lower, the higher the chance of sending correction
)
LED_MARTY_UPDATE_S = 0.8


class HeadMaster:
    def __init__(
        self, current_config, pose_duration=POSE_DURATION_S, logging_level=logging.INFO
    ):
        self.config = load_toml(CONFIG_FILE)
        self.current_config = current_config
        self.camera = self.init_camera(self.current_config["camera"])
        self.marty = self.init_marty()
        self.voice = self.init_voice()
        if self.marty:
            self.marty.init_generated_text(self.voice.generated_text_callback)
        self.pose_name = None
        self.pose_duration = pose_duration  # seconds
        self.poses = {
            pose_name: load_toml(os.path.join(POSES_FOLDER, pose_name, "pose.toml"))
            for pose_name in POSES_LIST
        }
        self.actual_run = []
        self.history = []
        self.ongoing_mistakes = {}
        self.name_files = None

        self.landmarker = setup_landmarker(self.config["model_path"])

        self.pose_ended = True
        self.is_pose_ending = False
        self.max_error_margin = self.config["feedback"]["max_error_margin"]

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging_level)

    def init_camera(self, camera_index=0):
        return cv2.VideoCapture(camera_index)

    def init_marty(self):
        try:
            marty = MyMarty(self.current_config["marty"])
            return marty
        except Exception as e:
            print(f"Failed to initialize Marty: {e}", file=sys.stderr)
            return None

    def init_voice(self):
        if self.marty:

            def move_marty_callback(chunk_duration):
                self.marty.move_marty_randomly(chunk_duration)

            def move_marty_callback_correctiv():
                self.marty.move_marty_limb()

        else:
            move_marty_callback = None
            move_marty_callback_correctiv = None

        return Speak(
            move_marty_callback,
            True,
            move_marty_callback_correctiv,
            self.analyze_ongoing_frame,
            can_i_speak=lambda: self.pose_ended or not self.is_pose_ending,
        )

    def filter_image(self, image):
        return apply_film_effect(image.numpy_view(), self.config["film_settings"])

    def update_ongoing_frame(self, elapsed):
        if len(self.actual_run) == 0:
            return
        frame = self.actual_run[-1]
        for angle_name, angle_data in frame.items():
            if (
                abs(angle_data["error"])
                < self.max_error_margin
                * SEND_CORRECTION_THRESHOLD
            ):
                if (
                    angle_name in self.ongoing_mistakes
                    and len(self.ongoing_mistakes[angle_name]["mistakes"][-1]) < 2
                ):
                    self.ongoing_mistakes[angle_name]["mistakes"][-1].append(elapsed)
                    self.ongoing_mistakes[angle_name]["timed_mistake"] += (
                        self.ongoing_mistakes[angle_name]["mistakes"][-1][1]
                        - self.ongoing_mistakes[angle_name]["mistakes"][-1][0]
                    )
                continue
            if angle_name not in self.ongoing_mistakes:
                self.ongoing_mistakes[angle_name] = {
                    "mistakes_repetitions": 0,
                    "timed_mistake": 0,
                    "remider_done": 0,
                    "target_angle": angle_data["target_angle"],
                    "current_angle": angle_data["current_angle"],
                    "mistakes": [[elapsed]],
                }
            if len(self.ongoing_mistakes[angle_name]["mistakes"][-1]) > 1:
                self.ongoing_mistakes[angle_name]["mistakes"].append([elapsed])
                self.ongoing_mistakes[angle_name]["mistakes_repetitions"] += 1

    def analyze_ongoing_frame(self):
        correction_to_do = {}
        for angle_name, mistakes in self.ongoing_mistakes.items():
            if len(mistakes["mistakes"][-1]) < 2:
                correction_to_do[
                    angle_name + " target:" + str(round(mistakes["target_angle"]))
                ] = "current:" + str(round(mistakes["current_angle"]))
                mistakes["remider_done"] += 1
        return correction_to_do

    def update_correction_feedback(self):
        if self.is_pose_ending or not self.voice.is_done():
            return
        correction = self.analyze_ongoing_frame()
        if bool(correction):
            print(correction)
            self.voice.correction = correction
            self.voice.move_marty_type_correctiv = correction
            self.voice.corrective_feedback(correction, self.poses[self.pose_name])

    def update_window(self, show_landmarks=False, timer_text="", elapsed=0.0):
        self.process_image(show_landmarks, timer_text, elapsed)
        if elapsed > MARGIN_BEFORE_CORRECTION_FEEDBACK_S:
            self.update_correction_feedback()

    def process_image(self, show_landmarks=False, timer_text="", elapsed=0.0):
        camera_image = capture_image_from_camera(self.camera)
        output_frame = self.filter_image(camera_image)
        if show_landmarks:
            result = self.analyze_image(camera_image)
            if result.pose_landmarks:
                self.actual_run.append(
                    draw_skeleton(
                        output_frame,
                        result.pose_landmarks,
                        self.config,
                        self.poses[self.pose_name]["pose"],
                        self.name_files,
                        self.marty,
                    )
                )
            self.update_ongoing_frame(elapsed)
            cv2.putText(
                output_frame,
                timer_text,
                (output_frame.shape[1] - 250, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        self.draw_overlays(output_frame)

        frame = cv2.resize(
            output_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        cv2.imshow("Video Feedback", frame)
        self.name_files = None

    def draw_overlays(self, frame):
        if self.voice.generated_text:
            cv2.rectangle(
                frame,
                (5, frame.shape[0] - 40),
                (frame.shape[1] - 5, frame.shape[0] - 5),
                (0, 0, 0),
                -1,
            )
            # Show only last 18 words
            showed_text = " ".join(self.voice.generated_text.split()[-18:])
            cv2.putText(
                frame,
                showed_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        if True:  # Debug mode
            cv2.rectangle(
                frame,
                (5, 5),
                (600, 40),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                f"Voice Queue: {self.voice.text_queue.qsize()}, Audio Queue: {self.voice.audio_queue.qsize()}, request Queue: {self.voice.request_queue.qsize()}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def analyze_image(self, image):
        timestamp = int(time.time() * 1000)
        return self.landmarker.detect_for_video(image, timestamp)

    def do_exercise(self, pose: str):
        self.load_pose(pose)
        feedback = self.do_pose()
        print()
        print("Feedback:", feedback)

    def wait(self):
        """Waits until all voice tasks are done, updating GUI."""
        while not self.voice.is_done():
            self.update_window(show_landmarks=False)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def wait_for_event(self, event):
        """
        Waits for a specific event (utterance completion),
        while keeping the GUI updated.
        """
        while not event.is_set():
            self.update_window(show_landmarks=False)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def reset_marty_pos(self):
        if self.marty:
            self.marty.load_and_do_pose(POSES_FOLDER + "mountain/pose.toml")

    def load_pose(self, pose):
        self.pose_name = pose

        intro_done_event = self.voice.load_pose(self.poses[pose])

        explanation_done_event = self.voice.show_pose(self.poses[pose])
        self.wait_for_event(intro_done_event)
        self.voice.move_marty_enabled = False
        if self.marty:
            self.marty.load_and_do_pose(POSES_FOLDER + pose + "/pose.toml")

        self.wait_for_event(explanation_done_event)
        self.reset_marty_pos()
        counter_done_event = self.voice.start_counter()
        self.wait_for_event(counter_done_event)
        self.voice.move_marty_enabled = True

    def do_pose(self):
        start_time = time.time()
        timer_text = ""
        self.actual_run = []
        self.ongoing_mistakes = {}
        elapsed_time = 0.0
        self.is_pose_ending = False
        self.pose_ended = False
        led_update_marty = 0.0
        while elapsed_time < self.pose_duration:
            elapsed_time = time.time() - start_time
            timer_text = f"Time in pose: {int(elapsed_time)}s / {self.pose_duration}s"

            if (
                not self.is_pose_ending
                and elapsed_time > self.pose_duration - TIME_GENERATION_END_FEEDBACK_S
            ):
                self.is_pose_ending = True
                feedbacks = get_feedbacks_from_run(
                    self.actual_run,
                    elapsed_time,
                    self.max_error_margin,
                )

                feedbacks["pose_name"] = self.pose_name
                feedback_dump = toml.dumps(feedbacks)
                print(feedback_dump)
                time.sleep(0.1)
                self.voice.end_pose_feedback(feedback_dump)

            self.update_window(
                show_landmarks=True,
                timer_text=timer_text,
                elapsed=elapsed_time,
            )
            if self.marty and self.marty.is_empty and elapsed_time > led_update_marty + LED_MARTY_UPDATE_S :
                mean_error = self.max_error_margin
                if len(self.actual_run) > 0:
                    errors = [val["error"] for val in self.actual_run[-1].values()]
                    mean_error = sum(errors) / len(errors) if errors else 0
                b, g, r = get_color_gradient(mean_error, self.max_error_margin)
                hex = f"#{r:02X}{g:02X}{b:02X}"
                self.marty.set_light_marty(hex, elapsed_time, self.pose_duration)
                led_update_marty = elapsed_time
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self.name_files = str(time.time())

        if self.marty:
            self.marty.disco_off()
        self.pose_ended = True
        return self.voice.generated_text

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()
