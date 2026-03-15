import sys

import cv2
import numpy as np


def load_pose_image_for_detection(pose_path, pose_name, image_name="original.png"):
    """Load a pose asset and normalize RGBA inputs for reliable landmark detection."""
    try:
        image = cv2.imread(pose_path, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        print(f"Error loading pose image: {exc}")
        return None

    if image is None:
        print(f"Could not read image at {pose_path}")
        return None

    # If the source has transparency, composite onto white so we don't lose content.
    if len(image.shape) == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        composited = rgb * alpha[..., None] + white * (1.0 - alpha[..., None])
        image = composited.astype(np.uint8)

    # Guardrail: alpha-only masks become black images and are poor pose inputs.
    if np.max(image) == 0:
        print(
            f"Warning: {pose_name}/{image_name} appears fully black after loading; "
            "landmark detection will likely fail.",
            file=sys.stderr,
        )

    return image
