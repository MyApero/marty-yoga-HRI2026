import numpy as np
import tomli


def load_toml(file_path):
    with open(file_path, "rb") as f:
        return tomli.load(f)


def get_angle(p1, p2, p3):
    """Calculates the angle at p2 given points p1, p2, and p3."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return int(angle)


def get_lerp_color(current, target, margin):
    """Calculates color from Green (0 err) to Red (margin err)"""
    error = abs(current - target)
    factor = min(error / margin, 1.0)
    r = int(255 * factor)
    g = int(255 * (1 - factor))
    return (0, g, r)
