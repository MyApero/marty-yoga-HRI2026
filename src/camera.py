import mediapipe as mp
import cv2


def _rotate_frame(frame, rotation_degrees):
    normalized = int(rotation_degrees) % 360
    if normalized == 0:
        return frame
    if normalized == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(
        f"Unsupported camera rotation: {rotation_degrees}. Use 0, 90, 180, or 270."
    )


def capture_image_from_camera(camera, rotation_degrees=0):
    success, frame = camera.read()
    if not success:
        raise RuntimeError("Failed to read from camera.")

    frame = cv2.flip(frame, 1)
    frame = _rotate_frame(frame, rotation_degrees)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return mp_image
