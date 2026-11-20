import cv2
import os
import time
import logging
import sys
import queue
import random
import threading
import toml
from src.utils import loading_print, get_color_gradient

loading_print("  Loading MediaPipe...")
import mediapipe as mp

from src.utils import load_toml
from src.mediapipe_operations import setup_landmarker
from src.live_pose import LivePose
from src.feedback_preprocess import get_feedbacks_from_run
from src.feedback_engine import FeedbackEngine
from src.session_state import SessionState
from src.marty import MyMarty
from src.camera import capture_image_from_camera
from src.window import WindowRenderer
from src.pose_image_loader import load_pose_image_for_detection
from enum import Enum

# Initial Setup
CONFIG_FILE = "config.toml"
LUCY_ROBOTS_FILE = "lucy_robots.toml"
POSES_FOLDER = "poses/"
POSES_LIST = [
    "chair",
    "crescent_moon",
    "left_warrior1",
    "left_warrior2",
    "right_warrior1",
    "right_warrior2",
    "mountain",
]
EXERCISE_KEY_MAP = {
    ord("1"): "chair",
    ord("2"): "crescent_moon",
    ord("3"): "left_warrior1",
    ord("4"): "left_warrior2",
    ord("5"): "mountain",
    ord("6"): "right_warrior1",
    ord("7"): "right_warrior2",
}
MEDIAPIPE_KEY_MAP = {
    ord("q"): "chair",
    ord("w"): "crescent_moon",
    ord("e"): "left_warrior1",
    ord("r"): "left_warrior2",
    ord("t"): "mountain",
    ord("y"): "right_warrior1",
    ord("u"): "right_warrior2",
}
RANDOM_DEMO_POOL = ["chair", "crescent_moon", "left_warrior1", "left_warrior2"]
RANDOM_DEMO_COUNT = 1
# Robots limited to the poses made for them in Lucy (unitree_warrior2, reachy_montain, reachy_warrior2).
ROBOT_POSES = {"unitree_g1_urdf": ["right_warrior2"], "reachy_mini_so101_humanoid_urdf": ["mountain", "left_warrior2"]}
MARTY_ONLY_POSE_DURATION_S = 5
MEDIAPIPE_ONLY_POSE_DURATION_S = 1500

MARGIN_BEFORE_CORRECTION_FEEDBACK_S = 5
TIME_GENERATION_END_FEEDBACK_S = 12.0
POSE_DURATION_S = 25
SEND_CORRECTION_THRESHOLD = (
    0.7  # the lower, the higher the chance of sending correction
)
LED_MARTY_UPDATE_S = 0.8
PLUGIN_ID = "yoga_coach"
SPEAKING_SESSIONS = {"demo", "full_session", "intro"}  # they open with Marty talking, so the spinner has an end
STOP_SETTLE_S = 2.0  # the rest pose takes 2 s; the Stop button keeps its spinner until Marty is there
DEMO_PLUGIN_ID = "yoga_demo"


class SessionStopped(Exception):
    """Raised from the window loop when Stop is pressed, so a session unwinds from wherever it is."""


def pose_label(pose: str) -> str:
    return pose.replace("_", " ").title()


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
        self.camera_rotation_degrees = int(
            self.current_config.get("camera_rotation_degrees", 0)
        )
        self.camera = self.init_camera(self.current_config)
        self.marty = self.init_marty() if enable_marty else None
        self.session = SessionState()
        self.interaction_state = InteractionState.IDLE
        self.starting = False
        self.voice = self.init_voice(enable_voice=enable_voice)
        self.window_renderer = WindowRenderer(self.config, self.voice, self.logger)
        if self.marty:
            self.marty.init_generated_text(self.voice.generated_text_callback)
        self.stop_requested = threading.Event()
        self.plugin_commands = queue.Queue()
        self.session_label = None
        self.stopping = False
        self.plugins = self.init_plugins()
        self.mjpeg = self.init_mjpeg()  # TEMPORARY: quick MJPEG server, deleted with src/mjpeg_server.py
        self.pose_duration = pose_duration  # seconds
        self.poses = {
            pose_name: load_toml(os.path.join(POSES_FOLDER, pose_name, "pose.toml"))
            for pose_name in POSES_LIST
        }
        for name, pose in self.poses.items():
            pose["image"] = self.load_pose_image(name, "image.jpg")
        self.ongoing_mistakes = {}

        self.landmarker = setup_landmarker(self.config["model_path"])
        self.live_pose = LivePose(self.config["model_path"])
        self.pose_is_new = False

        self.max_error_margin = self.config["feedback"]["max_error_margin"]
        self.feedback_engine = FeedbackEngine(
            self.max_error_margin,
            send_correction_threshold=SEND_CORRECTION_THRESHOLD,
        )
        self.ongoing_mistakes = self.feedback_engine.ongoing_mistakes
        self._cleaned_up = False

        perf_config = self.config.get("performance", {})
        self.show_perf_overlay = perf_config.get("show_perf_overlay", False)
        self.fps_smoothing = perf_config.get("fps_smoothing", 0.2)
        self.smoothed_fps = None
        self.last_show_ms = 0.0
        self.ignore_escape_until = 0.0

    def set_interaction_state(self, new_state):
        if self.interaction_state == new_state:
            return
        self.logger.debug(
            "Interaction transition: %s -> %s",
            self.interaction_state.name,
            new_state.name,
        )
        self.interaction_state = new_state

    def init_camera(self, camera_config=None):
        if isinstance(camera_config, dict):
            camera_index = camera_config.get("camera", 0)
            camera_width = camera_config.get("camera_width")
            camera_height = camera_config.get("camera_height")
            camera_fps = camera_config.get("camera_fps")
        else:
            camera_index = 0 if camera_config is None else camera_config
            camera_width = None
            camera_height = None
            camera_fps = None

        camera = cv2.VideoCapture(camera_index)

        if camera_width:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_width))
        if camera_height:
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_height))
        if camera_fps:
            camera.set(cv2.CAP_PROP_FPS, float(camera_fps))

        return camera

    def init_marty(self):
        try:
            if "lucy" in self.current_config:
                from src.lucy_robot import LucyCoachRobot

                robots = load_toml(LUCY_ROBOTS_FILE)
                preferred = self.current_config.get("lucy_robot")
                return LucyCoachRobot(self.current_config["lucy"], preferred, robots.get(preferred, {}), robots=robots)
            marty = MyMarty(self.current_config["marty"])
            return marty
        except Exception as e:
            print(f"Failed to initialize Marty: {e}", file=sys.stderr)
            return None

    def init_mjpeg(self):
        # TEMPORARY: quick MJPEG server for a latency comparison, deleted with src/mjpeg_server.py.
        from src.mjpeg_server import MjpegServer

        try:
            return MjpegServer()
        except OSError as error:
            print(f"MJPEG server not started: {error}", file=sys.stderr)
            return None

    def init_plugins(self):
        ros = getattr(self.marty, "ros", None)
        if ros is None:
            return []
        from src.lucy_plugin import LucyPlugin

        return [
            LucyPlugin(ros, PLUGIN_ID, self.plugin_manifest, self.on_plugin_action),
            LucyPlugin(ros, DEMO_PLUGIN_ID, self.demo_manifest, self.on_plugin_action),
        ]

    def plugin_manifest(self):
        busy = self.session_label is not None
        if self.stopping:
            return {
                "label": "Yoga coach",
                "caption": "YOGA",
                "icon": "loading",
                "status": "Stopping",
                "active": False,
                "actions": [{"id": "stop", "label": "Stopping", "disabled": True}],
            }
        poses = [{"id": pose, "label": pose_label(pose)} for pose in self.supported_poses()]
        counts = range(1, len(self.demo_pool()) + 1)
        return {
            "label": "Yoga coach",
            "caption": "YOGA",
            "icon": "pose",
            "status": self.session_label or "Ready",
            "active": busy,
            "actions": [
                {
                    "id": "full_session",
                    "label": "Full session",
                    "disabled": busy,
                    "options": [{"id": str(n), "label": f"{n} pose{'s' if n > 1 else ''}"} for n in counts],
                },
                {"id": "coach", "label": "Coach one pose", "disabled": busy, "options": poses},
                {"id": "demonstrate", "label": "Marty demonstrates a pose", "disabled": busy, "options": poses},
                {"id": "intro", "label": "Intro only", "disabled": busy},
                {"id": "stop", "label": "Stop", "danger": True, "disabled": not busy},
            ],
        }

    def demo_manifest(self):
        busy = self.session_label is not None
        if busy and self.starting:
            return {
                "label": "Yoga demo",
                "caption": "STOP",
                "icon": "loading",
                "status": f"Starting: {self.session_label}, click to stop",
                "active": True,
                "actions": [{"id": "stop", "label": "Stop", "danger": True}],
            }
        if self.stopping:
            return {
                "label": "Yoga demo",
                "caption": "STOPPING",
                "icon": "loading",
                "status": "Stopping",
                "active": False,
                "actions": [{"id": "stop", "label": "Stopping", "disabled": True}],
            }
        return {
            "label": "Yoga demo",
            "caption": "STOP" if busy else "DEMO",
            "icon": "stop" if busy else "play",
            "status": f"{self.session_label}, click to stop" if busy else "Click to play the demo",
            "active": busy,
            "actions": [{"id": "stop", "label": "Stop", "danger": True} if busy else {"id": "demo", "label": "Demo"}],
        }

    def on_plugin_action(self, action, option):
        # ROS thread: Stop takes effect at once, the rest waits for the main loop (OpenCV lives there).
        if action == "stop":
            if self.session_label is not None and not self.stopping:
                self.stopping = True
                self.stop_requested.set()
                self.announce()
            return
        if self.session_label is not None or self.stopping or not self.plugin_commands.empty():
            return  # a double click must not queue a second session that starts right after Stop
        self.plugin_commands.put((action, option))

    def announce(self):
        for plugin in self.plugins:
            plugin.announce()

    def supported_poses(self):
        return ROBOT_POSES.get(getattr(self.marty, "robot_package", None), list(EXERCISE_KEY_MAP.values()))

    def demo_pool(self):
        return ROBOT_POSES.get(getattr(self.marty, "robot_package", None), RANDOM_DEMO_POOL)

    def session_for(self, action, option=None):
        """(status label, run) for a session request, or None when the request names no session."""
        pool = self.demo_pool()
        if action == "intro":
            return "Intro", self.intro
        if action == "demo":
            poses = random.sample(pool, min(RANDOM_DEMO_COUNT, len(pool)))
            return "Demo", lambda: self.run_demo(poses=poses)
        if action == "full_session" and option and option.isdigit():
            count = int(option)
            if 1 <= count <= len(pool):
                poses = random.sample(pool, count)
                return f"Full session ({count} pose{'s' if count > 1 else ''})", lambda: self.run_demo(poses=poses)
        if option in self.supported_poses():
            if action == "coach":
                return f"Coaching {pose_label(option)}", lambda: self.do_mediapipe_pose_only(option)
            if action == "demonstrate":
                return f"Demonstrating {pose_label(option)}", lambda: self.do_marty_pose_only(option)
        return None

    def start(self, action, option=None):
        """Run one session to its end or until Stop; the keyboard and Lucy's plugin menu both land here."""
        session = self.session_for(action, option)
        if session is None:
            return
        self.session_label, run = session
        self.starting = self.voice is not None and action in SPEAKING_SESSIONS
        self.stop_requested.clear()
        self.announce()
        try:
            run()
        except SessionStopped:
            self.stop_session()
        finally:
            self.session_label = None
            self.starting = False
            self.stopping = False
            self.stop_requested.clear()
            self.announce()

    def on_voice_played(self):
        # Player thread: the first sound of a session ends the start spinner.
        if self.starting:
            self.starting = False
            self.announce()

    def stop_session(self):
        self.starting = False
        self.stopping = True
        with self.plugin_commands.mutex:
            self.plugin_commands.queue.clear()
        self.stop_requested.clear()
        self.announce()
        if self.voice:
            self.voice.stop()
            self.voice.move_marty_enabled = True
            self.voice.move_marty_type_corrective = None
        if self.marty:
            self.marty.stop_motion()
            self.marty.disco_off()
            self.reset_marty_pos()
        self.session.pose_ended = True
        self.session.is_pose_ending = False
        self.set_interaction_state(InteractionState.IDLE)
        if self.marty:
            self.settle(STOP_SETTLE_S)

    def settle(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.update_window(show_landmarks=False)
            cv2.waitKey(1)

    def run_plugin_command(self):
        try:
            action, option = self.plugin_commands.get_nowait()
        except queue.Empty:
            return
        self.start(action, option)

    def intro(self):
        self.wait_for_event(self.voice.intro())

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
            show_voice=getattr(self.marty, "show_voice", None),
            on_play=self.on_voice_played,
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

    def update_window(
        self,
        show_landmarks=False,
        timer_text="",
        elapsed=0.0,
        enable_corrective_feedback=True,
    ):
        if self.stop_requested.is_set():
            raise SessionStopped()
        self.process_camera_image(show_landmarks, timer_text, elapsed)
        if enable_corrective_feedback and elapsed > MARGIN_BEFORE_CORRECTION_FEEDBACK_S:
            self.update_correction_feedback()

    def process_camera_image(self, show_landmarks=False, timer_text="", elapsed=0.0):
        frame_start = time.perf_counter()

        capture_start = time.perf_counter()
        camera_image = capture_image_from_camera(
            self.camera,
            rotation_degrees=self.camera_rotation_degrees,
        )
        capture_ms = (time.perf_counter() - capture_start) * 1000.0

        frame = self.process_image(camera_image, show_landmarks, timer_text, elapsed)

        total_pre_show_ms = (time.perf_counter() - frame_start) * 1000.0
        if self.show_perf_overlay:
            renderer_timings = self.window_renderer.last_frame_timings
            total_ms = total_pre_show_ms + self.last_show_ms
            inst_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            if self.smoothed_fps is None:
                self.smoothed_fps = inst_fps
            else:
                alpha = max(0.01, min(1.0, float(self.fps_smoothing)))
                self.smoothed_fps = alpha * inst_fps + (1.0 - alpha) * self.smoothed_fps

            debug_lines = [
                f"FPS: {self.smoothed_fps:.1f}",
                f"Total: {total_ms:.1f} ms",
                f"Capture: {capture_ms:.1f} ms",
                f"Filter: {renderer_timings.get('filter_ms', 0.0):.1f} ms",
                f"Detect: {renderer_timings.get('detect_ms', 0.0):.1f} ms",
                f"Resize: {renderer_timings.get('resize_ms', 0.0):.1f} ms",
                f"Render: {renderer_timings.get('render_ms', 0.0):.1f} ms",
                f"Show(prev): {self.last_show_ms:.1f} ms",
            ]
            self._draw_perf_overlay(frame, debug_lines)

        show_start = time.perf_counter()
        self.window_renderer.show(frame)
        if hasattr(self.marty, "publish_frame"):
            self.marty.publish_frame(frame)
        if self.mjpeg:  # TEMPORARY: quick MJPEG server, deleted with src/mjpeg_server.py
            self.mjpeg.push(frame)
        self.last_show_ms = (time.perf_counter() - show_start) * 1000.0
        self.session.name_files = None

    def _draw_perf_overlay(self, frame, lines):
        if not lines:
            return

        margin = 10
        line_height = 20
        box_width = 280
        box_height = line_height * len(lines) + 12

        top_left = (frame.shape[1] - box_width - margin, margin)
        bottom_right = (frame.shape[1] - margin, margin + box_height)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)

        y = margin + 20
        x = frame.shape[1] - box_width
        for line in lines:
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += line_height

    def process_image(
        self, image, show_landmarks=False, timer_text="", elapsed=0.0, show_state=False, analyze=None
    ):
        pose_data = (
            self.poses.get(self.session.pose_name) if self.session.pose_name else None
        )
        if show_state:
            interaction_state_text = self.interaction_state.name.replace(
                "_", " "
            ).title()
        else:
            interaction_state_text = None
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
            analyze_image=analyze or self.analyze_image,
        )

        if show_landmarks:
            if frame_angles is not None and self.pose_is_new:
                self.session.actual_run.append(frame_angles)
            self.update_ongoing_frame(elapsed)

        return frame

    def analyze_image(self, image):
        # The live loop draws the latest pose; each result joins the run once, when it is new.
        result, self.pose_is_new = self.live_pose.analyze(image)
        return result

    def analyze_still(self, image):
        self.pose_is_new = True
        return self.landmarker.detect_for_video(image, int(time.time() * 1000))

    def do_exercise(self, pose: str):
        self.load_pose(pose)
        feedback = self.do_pose()
        print()
        print("Feedback:", feedback)

    def do_marty_pose_only(self, pose: str, duration_s=MARTY_ONLY_POSE_DURATION_S):
        self.session.pose_name = pose
        self.session.pose_ended = False
        self.set_interaction_state(InteractionState.PRESENTING)

        if self.marty:
            self.marty.load_and_do_pose(POSES_FOLDER + pose + "/pose.toml")

        start_time = time.time()
        elapsed_time = 0.0
        while elapsed_time < duration_s:
            elapsed_time = time.time() - start_time
            timer_text = f"Hold: {int(elapsed_time)}s / {duration_s}s"
            self.update_window(
                show_landmarks=False, timer_text=timer_text, elapsed=elapsed_time
            )
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        self.reset_marty_pos()
        self.session.pose_ended = True
        self.set_interaction_state(InteractionState.IDLE)

    def do_mediapipe_pose_only(
        self, pose: str, duration_s=MEDIAPIPE_ONLY_POSE_DURATION_S
    ):
        self.session.pose_name = pose
        self.session.pose_ended = False
        self.session.is_pose_ending = False
        self.session.actual_run = []
        self.feedback_engine.reset()
        self.session.pose_name = pose
        self.session.pose_ended = False
        self.set_interaction_state(InteractionState.PRESENTING)

        if self.marty:
            self.marty.load_and_do_pose(POSES_FOLDER + pose + "/pose.toml")

        start_time = time.time()
        elapsed_time = 0.0
        while elapsed_time < duration_s:
            elapsed_time = time.time() - start_time
            timer_text = f"Time in pose: {int(elapsed_time)}s / {duration_s}s"
            self.update_window(
                show_landmarks=True,
                timer_text=timer_text,
                elapsed=elapsed_time,
                enable_corrective_feedback=False,
            )
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                # Avoid using the same ESC keypress to trigger global quit right after abort.
                self.ignore_escape_until = time.time() + 0.35
                break
            if key == ord("c"):
                self.session.name_files = str(time.time())

        self.session.pose_ended = True
        self.set_interaction_state(InteractionState.IDLE)

    def wait_for_event(self, event):
        """
        Waits for a specific event (utterance completion),
        while keeping the GUI updated.
        """
        while not event.is_set():
            self.update_window(show_landmarks=False)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
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
            if key == 27:
                break
            if key == ord("c"):
                self.session.name_files = str(time.time())

        if self.marty:
            self.marty.disco_off()
        self.session.pose_ended = True
        self.set_interaction_state(InteractionState.IDLE)
        return self.voice.generated_text

    def load_pose_image(self, pose_name, image_name="original.png"):
        pose_path = os.path.join(POSES_FOLDER, pose_name, image_name)
        return load_pose_image_for_detection(pose_path, pose_name, image_name)

    def generate_yoga_images_with_landmarks(
        self,
        poses=[
            "chair",
            "crescent_moon",
            "left_warrior1",
            "right_warrior1",
            "left_warrior2",
            "right_warrior2",
            "mountain",
        ],
        verbose=True,
    ):
        saved_files = []
        for pose_name in poses:
            self.session.pose_name = pose_name
            self.session.name_files = f"pose_{pose_name}_{int(time.time())}"
            image_bgr = self.load_pose_image(pose_name)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Helpful diagnostics for generation mode.
            result = self.analyze_still(mp_image)
            if not result.pose_landmarks and verbose:
                print(
                    f"Warning: no landmarks detected for pose '{pose_name}'. "
                    "This input image may be too stylized or mask-like.",
                    file=sys.stderr,
                )

            frame = self.process_image(mp_image, show_landmarks=True, analyze=self.analyze_still)
            output_filename = f"output_{pose_name}.jpg"
            cv2.imwrite(output_filename, frame)
            saved_files.append(output_filename)
            if verbose:
                print(f"Saved {output_filename}")
        return saved_files

    def run_demo(self, poses=None):
        if poses is None:
            poses = [
                "chair",
                "crescent_moon",
                "left_warrior1",
                "left_warrior2",
                "mountain",
            ]
        self.voice.intro()
        for pose in poses:
            self.do_exercise(pose)
        self.wait_for_event(self.voice.outro())  # the demo, and its Stop button, last until Marty says goodbye

    def handle_key_event(self, key):
        if key == 27:
            if time.time() < self.ignore_escape_until:
                return True
            self.cleanup()
            return False
        if key == ord("s"):
            self.start("intro")
        elif key == ord("p"):
            self.logger.setLevel(logging.WARNING)
            self.generate_yoga_images_with_landmarks()
        elif key == ord("d"):
            self.start("demo")
        elif key in EXERCISE_KEY_MAP:
            self.start("demonstrate", EXERCISE_KEY_MAP[key])
        elif key in MEDIAPIPE_KEY_MAP:
            self.start("coach", MEDIAPIPE_KEY_MAP[key])
        return True

    def tick(self, key):
        keep_running = self.handle_key_event(key)
        if not keep_running:
            return False
        self.run_plugin_command()
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
