import logging
import cv2

from src.mediapipe_operations import apply_film_effect
from src.video_feedback import draw_skeleton


class WindowRenderer:
    def __init__(self, config, voice, logger):
        self.config = config
        self.voice = voice
        self.logger = logger

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
        name_file=None,
        marty=None,
        analyze_image=None,
    ):
        output_frame = self.filter_image(image)
        frame_angles = None

        if show_landmarks and analyze_image and pose_data is not None:
            result = analyze_image(image)
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

        self.draw_overlays(output_frame, pose_name, pose_data, pose_ended)

        frame = cv2.resize(
            output_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        return frame, frame_angles

    def draw_overlays(self, frame, pose_name, pose_data, pose_ended):
        if self.voice.subtitles:
            cv2.rectangle(
                frame,
                (5, frame.shape[0] - 40),
                (frame.shape[1] - 5, frame.shape[0] - 5),
                (0, 0, 0),
                -1,
            )
            # Show only last 18 words
            showed_text = " ".join(self.voice.subtitles.split()[-18:])
            cv2.putText(
                frame,
                showed_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        if self.logger.isEnabledFor(logging.DEBUG):
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
