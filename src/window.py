import logging
import cv2
import time

from src.mediapipe_operations import apply_film_effect
from src.video_feedback import draw_skeleton


class WindowRenderer:
    def __init__(self, config, voice, logger):
        self.config = config
        self.voice = voice
        self.logger = logger
        self.last_frame_timings = {
            "filter_ms": 0.0,
            "detect_ms": 0.0,
            "resize_ms": 0.0,
            "render_ms": 0.0,
        }

    def show(self, frame):
        cv2.imshow("Video Feedback", frame)

    def filter_image(self, image):
        return apply_film_effect(image.numpy_view(), self.config["film_settings"])

    def process_image(
        self,
        image,
        show_landmarks=False,
        timer_text="",
        pose_name=None,
        pose_data=None,
        pose_ended=True,
        interaction_state_text=None,
        name_file=None,
        marty=None,
        analyze_image=None,
    ):
        frame_start = time.perf_counter()

        filter_start = time.perf_counter()
        output_frame = self.filter_image(image)
        filter_ms = (time.perf_counter() - filter_start) * 1000.0

        frame_angles = None
        detect_ms = 0.0

        if show_landmarks and analyze_image and pose_data is not None:
            detect_start = time.perf_counter()
            result = analyze_image(image)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
            if result.pose_landmarks:
                frame_angles = draw_skeleton(
                    output_frame,
                    result.pose_landmarks,
                    self.config,
                    pose_data["pose"],
                    name_file,
                    marty,
                )
            cv2.putText(
                output_frame,
                timer_text,
                (output_frame.shape[1] - 350, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        self.draw_overlays(
            output_frame,
            pose_name,
            pose_data,
            pose_ended,
            interaction_state_text,
        )

        resize_start = time.perf_counter()
        frame = cv2.resize(
            output_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

        resize_ms = (time.perf_counter() - resize_start) * 1000.0
        render_ms = (time.perf_counter() - frame_start) * 1000.0
        self.last_frame_timings = {
            "filter_ms": filter_ms,
            "detect_ms": detect_ms,
            "resize_ms": resize_ms,
            "render_ms": render_ms,
        }

        return frame, frame_angles

    def draw_overlays(
        self,
        frame,
        pose_name,
        pose_data,
        pose_ended,
        interaction_state_text=None,
    ):
        subtitles = self.voice.subtitles if self.voice else ""
        if subtitles:
            cv2.rectangle(
                frame,
                (5, frame.shape[0] - 40),
                (frame.shape[1] - 5, frame.shape[0] - 5),
                (0, 0, 0),
                -1,
            )
            # Show only last 18 words
            showed_text = " ".join(subtitles.split()[-18:])
            cv2.putText(
                frame,
                showed_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        if interaction_state_text:
            state_label = f"State: {interaction_state_text}"
            state_font = cv2.FONT_HERSHEY_SIMPLEX
            state_scale = 0.6
            state_thickness = 2
            text_size, baseline = cv2.getTextSize(
                state_label,
                state_font,
                state_scale,
                state_thickness,
            )
            text_width, text_height = text_size
            margin = 10

            # Keep state text above the subtitle area.
            text_x = frame.shape[1] - text_width - margin
            text_y = frame.shape[0] - 48

            rect_top_left = (text_x - 8, text_y - text_height - 6)
            rect_bottom_right = (text_x + text_width + 8, text_y + baseline + 6)
            cv2.rectangle(
                frame,
                rect_top_left,
                rect_bottom_right,
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                state_label,
                (text_x, text_y),
                state_font,
                state_scale,
                (255, 255, 255),
                state_thickness,
            )

        if self.logger.isEnabledFor(logging.DEBUG):
            text_queue_size = self.voice.text_queue.qsize() if self.voice else 0
            audio_queue_size = self.voice.audio_queue.qsize() if self.voice else 0
            request_queue_size = self.voice.request_queue.qsize() if self.voice else 0
            cv2.rectangle(
                frame,
                (5, 5),
                (600, 40),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                f"Voice Queue: {text_queue_size}, Audio Queue: {audio_queue_size}, request Queue: {request_queue_size}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        if self.logger.isEnabledFor(logging.INFO):
            if (
                pose_name
                and pose_data is not None
                and pose_data["image"] is not None
                and not pose_ended
            ):
                image_size = 300
                pose_image_resized = cv2.resize(
                    pose_data["image"],
                    (image_size, image_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                frame[10 : 10 + image_size, 10 : 10 + image_size] = pose_image_resized
                full_pose_name = pose_name.replace("_", " ").title()
                yoga_name = pose_data.get("yoga_name", "")
                if yoga_name:
                    full_pose_name += f" ({yoga_name})"
                cv2.putText(
                    frame,
                    full_pose_name,
                    (10, image_size + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
