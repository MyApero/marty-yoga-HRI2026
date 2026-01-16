
import cv2
import numpy as np
from src.utils import get_angle

def draw_skeleton(image, pose_landmarks, config, targets):
    h, w, _ = image.shape
    f_settings = config["film_settings"]
    s_settings = config["skeleton"]
    fb_settings = config["feedback"]
    default_color = tuple(config["colors"].get("default_skeleton", [255, 255, 255]))

    # Margin for error (10 degrees as you requested)
    margin = fb_settings.get("max_error_margin", 10.0)

    for pose in pose_landmarks:
        # --- 1. COORDINATE PREP ---
        sh_center_x = (pose[11].x + pose[12].x) / 2
        sh_center_y = (pose[11].y + pose[12].y) / 2
        hip_center_x = (pose[23].x + pose[24].x) / 2
        hip_center_y = (pose[23].y + pose[24].y) / 2

        pt_sh_center = (int(sh_center_x * w), int(sh_center_y * h))
        pt_hip_center = (int(hip_center_x * w), int(hip_center_y * h))
        pt_nose = (int(pose[0].x * w), int(pose[0].y * h))

        # --- 2. PRE-CALCULATE ANGLES AND COLORS ---
        # We store the color results so limbs can look them up
        joint_status = {}
        for joint in fb_settings["joints_to_monitor"]:
            p1, p2, p3 = (
                pose[joint["points"][0]],
                pose[joint["points"][1]],
                pose[joint["points"][2]],
            )

            if all(p.presence > 0.5 for p in [p1, p2, p3]):
                angle = get_angle(p1, p2, p3)
                target = targets.get(
                    joint["name"], angle
                )  # Fallback to current if not in TOML

                # LINEAR COLOR CALCULATION (LERP)
                error = abs(angle - target)
                factor = min(error / margin, 1.0)

                # BGR: Green (0, 255, 0) to Red (0, 0, 255)
                # factor 0 = Green, factor 1 = Red
                b = 0
                g = int(255 * (1 - factor))
                r = int(255 * factor)

                joint_status[joint["name"]] = {
                    "angle": angle,
                    "color": (b, g, r),
                    "pos": (int(p2.x * w), int(p2.y * h)),
                }

        # --- 3. THE SPINE (Nose -> Shoulders -> Hips) ---
        # Draws in default color as it's the anchor
        cv2.line(
            image, pt_nose, pt_sh_center, default_color, f_settings["line_thickness"]
        )
        cv2.line(
            image,
            pt_sh_center,
            pt_hip_center,
            default_color,
            f_settings["line_thickness"],
        )

        # --- 4. THE DYNAMIC HEAD ---
        ear_l, ear_r = pose[7], pose[8]
        if ear_l.presence > 0.5 and ear_r.presence > 0.5:
            dist = np.sqrt((ear_l.x - ear_r.x) ** 2 + (ear_l.y - ear_r.y) ** 2)
            radius = int((dist * w) * f_settings["head_radius_multiplier"])
            cv2.circle(image, pt_nose, radius, default_color, 2)

        # --- 5. ARMS AND LEGS WITH DYNAMIC COLOR ---
        # Map limbs to their controlling joint for color feedback
        limb_to_joint = {
            13: "L-Elbow",
            15: "L-Elbow",
            17: "L-Elbow",
            19: "L-Elbow",
            21: "L-Elbow",
            14: "R-Elbow",
            16: "R-Elbow",
            18: "R-Elbow",
            20: "R-Elbow",
            22: "R-Elbow",
            25: "L-Knee",
            27: "L-Knee",
            23: "L-Hip",
            26: "R-Knee",
            28: "R-Knee",
            24: "R-Hip",
        }

        for start_idx, end_idx in s_settings["body_lines"] + s_settings["hand_lines"]:
            start, end = pose[start_idx], pose[end_idx]

            if start.presence > 0.5 and end.presence > 0.5:
                # Decide color: if the end-part of the limb is a monitored joint, use its color
                # Otherwise, use the default film color
                j_name = limb_to_joint.get(end_idx)
                line_color = (
                    joint_status[j_name]["color"]
                    if j_name in joint_status
                    else default_color
                )

                if start_idx in [11, 12]:
                    p1 = pt_sh_center
                elif start_idx in [23, 24]:
                    p1 = pt_hip_center
                else:
                    p1 = (int(start.x * w), int(start.y * h))

                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(image, p1, p2, line_color, f_settings["line_thickness"])

        # --- 6. ANGLE FEEDBACK LABELS ---
        for name, data in joint_status.items():
            text_pos = (data["pos"][0] + 10, data["pos"][1] - 10)
            label = (
                f"{name}: {data['angle']}°"
                if fb_settings["show_labels"]
                else f"{data['angle']}°"
            )

            # Draw highlight at joint
            cv2.circle(image, data["pos"], 4, data["color"], -1)
            # Text uses the same dynamic color
            cv2.putText(
                image,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                fb_settings["text_scale"],
                data["color"],
                1,
            )

