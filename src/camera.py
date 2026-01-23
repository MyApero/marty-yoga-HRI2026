
import mediapipe as mp
import cv2

def capture_image_from_camera(camera):
    success, frame = camera.read()
    if not success:
        raise RuntimeError("Failed to read from camera.")

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return mp_image
