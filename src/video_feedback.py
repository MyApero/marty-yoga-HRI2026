import cv2
from src.utils import get_skeleton_coordinates, get_angles_error_from_landmarks, get_joint_color
from pathlib import Path
from .marty import MyMarty
import toml

def draw_skeleton(image, pose_landmarks, config, targets, name_file: str|None=None, marty:MyMarty|None=None):
    h, w, _ = image.shape
    joint_connections = config["skeleton"]["joint_connections"]
    fb_settings = config["feedback"]
    margin = fb_settings.get("max_error_margin", 10.0)

    if isinstance(pose_landmarks[0], list):
        actual_landmarks = pose_landmarks[0]
    else:
        actual_landmarks = pose_landmarks
    coord_map = get_skeleton_coordinates(actual_landmarks, w, h)
    angles = get_angles_error_from_landmarks(coord_map, targets, fb_settings["angles"])

    for connection in joint_connections:
        idx1, idx2 = connection["joint"]
        
        if idx1 in coord_map and idx2 in coord_map:
            line_color = get_joint_color(
                idx1, idx2, 
                angles, 
                joint_connections, 
                threshold=margin
            )
            
            cv2.line(
                image, 
                coord_map[idx1], 
                coord_map[idx2], 
                line_color, 
                3,
                cv2.LINE_AA
            )
            
            cv2.circle(image, coord_map[idx1], 5, line_color, -1)

        current_angle_value = None
        angle_name = ""

        for name, data in angles.items():
            for pts in fb_settings["angles"]:
                if pts["name"] == name and pts["points"][0][1] == idx1:
                    current_angle_value = data['current_angle']
                    angle_name = name
                    break
        
        if current_angle_value is not None:
            text_pos = (coord_map[idx1][0] + 10, coord_map[idx1][1] - 10)
            
            label = (
                f"{angle_name}: {int(current_angle_value)}deg"
                if fb_settings.get("show_labels", True)
                else f"{int(current_angle_value)}deg"
            )

            # cv2.putText(
            #     image,
            #     label,
            #     text_pos,
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     fb_settings.get("text_scale", 0.5), # 0.2 is very small, 0.5 is better
            #     line_color,
            #     2,
            #     cv2.LINE_AA
            # )

    if name_file:
        folder_path = Path("poses/" + name_file)
        folder_path.mkdir(parents=True, exist_ok=True)
        toml_file = folder_path / "pose.toml"
        marty_data = {
            "LeftHip":0,
            "LeftTwist":0,
            "LeftKnee":0,
            "RightHip":0,
            "RightTwist":0,
            "RightKnee":0,
            "LeftArm":0,
            "RightArm":0,
            "Eyes":0,
            }
        if marty:
            marty_data = marty.get_pose()

        data = {
            "pose": {
                name: float(angle_data["current_angle"])
                for name, angle_data in angles.items()
            },
            "marty": marty_data
        }
        with open(toml_file, "w", encoding="utf-8") as f:
            toml.dump(data, f)
        print(f"Pose data saved to {toml_file}")
    return angles
