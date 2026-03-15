import cv2
import os
import time
import logging
import sys
import toml
from src.utils import loading_print, get_color_gradient

loading_print("  Loading MediaPipe...")
import mediapipe as mp

from src.utils import load_toml
from src.mediapipe_operations import setup_landmarker
from src.feedback_preprocess import get_feedbacks_from_run
from src.feedback_engine import FeedbackEngine
from src.session_state import SessionState
from src.marty import MyMarty
from src.camera import capture_image_from_camera
from src.window import WindowRenderer
from enum import Enum

# Initial Setup
CONFIG_FILE = "config.toml"
POSES_FOLDER = "poses/"
POSES_LIST = ["right_warrior2", "left_warrior2", "chair", "mountain"]

MARGIN_BEFORE_CORRECTION_FEEDBACK_S = 5
TIME_GENERATION_END_FEEDBACK_S = 12.0
POSE_DURATION_S = 50
SEND_CORRECTION_THRESHOLD = (
    0.7  # the lower, the higher the chance of sending correction
)
LED_MARTY_UPDATE_S = 0.8


class InteractionState(Enum):
    IDLE = 0
    PRESENTING = 1
    EXPLAINING_POSE = 2
    COUNTDOWN = 3
    IN_POSE_NO_CORRECTIVE_FEEDBACK = 4
    IN_POSE_CORRECTIVE_FEEDBACK_GENERATION = 5
    IN_POSE_CORRECTIVE_FEEDBACK = 6
    IN_POSE_END_FEEDBACK_GENERATION = 7
    END_FEEDBACK = 8


class HeadMaster:
    def __init__(
        self,
        current_config,
        pose_duration=POSE_DURATION_S,
        logging_level=logging.INFO,
        enable_marty=True,
        enable_voice=True,
    ):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging_level)

        self.config = load_toml(CONFIG_FILE)
        self.current_config = current_config
        self.camera = self.init_camera(self.current_config["camera"])
        self.marty = self.init_marty() if enable_marty else None
        self.session = SessionState()
        self.interaction_state = InteractionState.IDLE
        self.voice = self.init_voice(enable_voice=enable_voice)
        self.window_renderer = WindowRenderer(self.config, self.voice, self.logger)
        if self.marty:
            self.marty.init_generated_text(self.voice.generated_text_callback)
        self.pose_duration = pose_duration  # seconds
        self.poses = {
            pose_name: load_toml(os.path.join(POSES_FOLDER, pose_name, "pose.toml"))
            for pose_name in POSES_LIST
        }
        for name, pose in self.poses.items():
            pose["image"] = self.load_pose_image(name, "image.jpg")
        self.ongoing_mistakes = {}

        self.landmarker = setup_landmarker(self.config["model_path"])

        self.max_error_margin = self.config["feedback"]["max_error_margin"]
        self.feedback_engine = FeedbackEngine(
            self.max_error_margin,
            send_correction_threshold=SEND_CORRECTION_THRESHOLD,
        )
        self.ongoing_mistakes = self.feedback_engine.ongoing_mistakes
        self._cleaned_up = False

    def set_interaction_state(self, new_state):
        if self.interaction_state == new_state:
            return
        self.logger.debug(
            "Interaction transition: %s -> %s",
            self.interaction_state.name,
            new_state.name,
        )
        self.interaction_state = new_state

    def init_camera(self, camera_index=0):
        return cv2.VideoCapture(camera_index)

    def init_marty(self):
        try:
            marty = MyMarty(self.current_config["marty"])
            return marty
        except Exception as e:
            print(f"Failed to initialize Marty: {e}", file=sys.stderr)
            return None

    def init_voice(self, enable_voice=True):
        if not enable_voice:
            return None

        from src.speak import Speak

        if self.marty:

            def move_marty_callback(chunk_duration):
                self.marty.move_marty_randomly(chunk_duration)

            def move_marty_callback_corrective():
                self.marty.move_marty_limb()

        else:
            move_marty_callback = None
            move_marty_callback_corrective = None

        return Speak(
            move_marty_callback,
            True,
            move_marty_callback_corrective,
            self.analyze_ongoing_frame,
            can_i_speak=lambda: (
                self.session.pose_ended or not self.session.is_pose_ending
            ),
        )

    def update_ongoing_frame(self, elapsed):
        self.feedback_engine.update_ongoing_frame(self.session.actual_run, elapsed)

    def analyze_ongoing_frame(self):
        return self.feedback_engine.analyze_ongoing_frame()

    def update_correction_feedback(self):
        if self.session.is_pose_ending or not self.voice.is_done():
            return
        self.set_interaction_state(
            InteractionState.IN_POSE_CORRECTIVE_FEEDBACK_GENERATION
        )
        correction = self.analyze_ongoing_frame()
        if bool(correction):
            self.set_interaction_state(InteractionState.IN_POSE_CORRECTIVE_FEEDBACK)
            self.voice.correction = correction
            self.voice.move_marty_type_corrective = correction
            self.voice.corrective_feedback(
                correction, self.poses[self.session.pose_name]
            )
        else:
            self.set_interaction_state(InteractionState.IN_POSE_NO_CORRECTIVE_FEEDBACK)

    def update_window(self, show_landmarks=False, timer_text="", elapsed=0.0):
        self.process_camera_image(show_landmarks, timer_text, elapsed)
        if elapsed > MARGIN_BEFORE_CORRECTION_FEEDBACK_S:
            self.update_correction_feedback()

    def process_camera_image(self, show_landmarks=False, timer_text="", elapsed=0.0):
        camera_image = capture_image_from_camera(self.camera)
        frame = self.process_image(camera_image, show_landmarks, timer_text, elapsed)
        self.window_renderer.show(frame)
        self.session.name_files = None

    def process_image(self, image, show_landmarks=False, timer_text="", elapsed=0.0):
        pose_data = (
            self.poses.get(self.session.pose_name) if self.session.pose_name else None
        )
        interaction_state_text = self.interaction_state.name.replace("_", " ").title()
        frame, frame_angles = self.window_renderer.process_image(
            image,
            show_landmarks=show_landmarks,
            timer_text=timer_text,
            pose_name=self.session.pose_name,
            pose_data=pose_data,
            pose_ended=self.session.pose_ended,
            interaction_state_text=interaction_state_text,
            name_file=self.session.name_files,
            marty=self.marty,
            analyze_image=self.analyze_image,
        )

        if show_landmarks:
            if frame_angles is not None:
                self.session.actual_run.append(frame_angles)
            self.update_ongoing_frame(elapsed)

        return frame

    def analyze_image(self, image):
        timestamp = int(time.time() * 1000)
        return self.landmarker.detect_for_video(image, timestamp)

    def do_exercise(self, pose: str):
        self.load_pose(pose)
        feedback = self.do_pose()
        print()
        print("Feedback:", feedback)

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
        self.set_interaction_state(InteractionState.PRESENTING)
        self.session.pose_name = pose

        intro_done_event = self.voice.load_pose(self.poses[pose])

        explanation_done_event = self.voice.show_pose(self.poses[pose])
        self.wait_for_event(intro_done_event)
        self.set_interaction_state(InteractionState.EXPLAINING_POSE)
        self.voice.move_marty_enabled = False
        if self.marty:
            self.marty.load_and_do_pose(POSES_FOLDER + pose + "/pose.toml")

        self.wait_for_event(explanation_done_event)
        self.reset_marty_pos()
        self.set_interaction_state(InteractionState.COUNTDOWN)
        counter_done_event = self.voice.start_counter()
        self.wait_for_event(counter_done_event)
        self.voice.move_marty_enabled = True

    def do_pose(self):
        self.set_interaction_state(InteractionState.IN_POSE_NO_CORRECTIVE_FEEDBACK)
        start_time = time.time()
        timer_text = ""
        self.session.actual_run = []
        self.feedback_engine.reset()
        elapsed_time = 0.0
        self.session.is_pose_ending = False
        self.session.pose_ended = False
        led_update_marty = 0.0
        while elapsed_time < self.pose_duration:
            elapsed_time = time.time() - start_time
            timer_text = f"Time in pose: {int(elapsed_time)}s / {self.pose_duration}s"

            if (
                not self.session.is_pose_ending
                and elapsed_time > self.pose_duration - TIME_GENERATION_END_FEEDBACK_S
            ):
                self.session.is_pose_ending = True
                self.set_interaction_state(
                    InteractionState.IN_POSE_END_FEEDBACK_GENERATION
                )
                feedbacks = get_feedbacks_from_run(
                    self.session.actual_run,
                    elapsed_time,
                    self.max_error_margin,
                )

                feedbacks["pose_name"] = self.session.pose_name
                feedback_dump = toml.dumps(feedbacks)
                # print(feedback_dump)
                time.sleep(0.1)
                self.set_interaction_state(InteractionState.END_FEEDBACK)
                self.voice.end_pose_feedback(feedback_dump)

            self.update_window(
                show_landmarks=True,
                timer_text=timer_text,
                elapsed=elapsed_time,
            )
            if (
                self.marty
                and self.marty.is_empty
                and elapsed_time > led_update_marty + LED_MARTY_UPDATE_S
            ):
                mean_error = self.max_error_margin
                if len(self.session.actual_run) > 0:
                    errors = [
                        val["error"] for val in self.session.actual_run[-1].values()
                    ]
                    mean_error = sum(errors) / len(errors) if errors else 0
                b, g, r = get_color_gradient(mean_error, self.max_error_margin)
                hex = f"#{r:02X}{g:02X}{b:02X}"
                self.marty.set_light_marty(hex, elapsed_time, self.pose_duration)
                led_update_marty = elapsed_time
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self.session.name_files = str(time.time())

        if self.marty:
            self.marty.disco_off()
        self.session.pose_ended = True
        self.set_interaction_state(InteractionState.IDLE)
        return self.voice.generated_text

    def load_pose_image(self, pose_name, image_name="image.jpg"):
        pose_path = os.path.join(POSES_FOLDER, pose_name, image_name)
        try:
            image = cv2.imread(pose_path)
        except Exception as e:
            print(f"Error loading pose image: {e}")
            image = None
        if image is None:
            print(f"Could not read image at {pose_path}")
        return image

    def generate_yoga_images_with_landmarks(
        self,
        poses=["chair", "left_warrior2", "mountain", "right_warrior2"],
        verbose=True,
    ):
        saved_files = []
        for pose_name in poses:
            self.session.pose_name = pose_name
            self.session.name_files = f"pose_{pose_name}_{int(time.time())}"
            image_bgr = self.load_pose_image(pose_name)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            frame = self.process_image(mp_image, show_landmarks=True)
            output_filename = f"output_{pose_name}.jpg"
            cv2.imwrite(output_filename, frame)
            saved_files.append(output_filename)
            if verbose:
                print(f"Saved {output_filename}")
        return saved_files

    def run_demo(self, poses=None):
        if poses is None:
            poses = ["chair", "left_warrior2"]
        self.voice.intro()
        for pose in poses:
            self.do_exercise(pose)

    def handle_key_event(self, key):
        if key == ord("q"):
            self.cleanup()
            return False
        if key == ord("s"):
            self.voice.intro()
        elif key == ord("y"):
            self.logger.setLevel(logging.WARNING)
            self.generate_yoga_images_with_landmarks()
        elif key == ord("d"):
            self.run_demo()
        return True

    def tick(self, key):
        keep_running = self.handle_key_event(key)
        if not keep_running:
            return False
        self.process_camera_image()
        return True

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self.marty:
            try:
                self.marty.disco_off()
            except Exception:
                pass

        if self.voice:
            try:
                self.voice.shutdown()
            except Exception:
                pass

        try:
            self.camera.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
