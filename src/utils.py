import numpy as np
import tomli


def load_toml(file_path):
    with open(file_path, "rb") as f:
        return tomli.load(f)


def interpolate_point(p1, p2, t):
    return (
        int(p1[0] + t * (p2[0] - p1[0])),
        int(p1[1] + t * (p2[1] - p1[1]))
    )


def get_skeleton_coordinates(landmarks, w, h):
    """Maps MP landmarks and calculates custom virtual landmarks."""
    coord_map = {}
    for i, lm in enumerate(landmarks):
        coord_map[i] = (int(lm.x * w), int(lm.y * h))

    # Mid-Shoulder
    if 11 in coord_map and 12 in coord_map:
        coord_map[998] = (
            (coord_map[11][0] + coord_map[12][0]) // 2,
            (coord_map[11][1] + coord_map[12][1]) // 2
        )
    
    # Mid-Hip
    if 23 in coord_map and 24 in coord_map:
        coord_map[997] = (
            (coord_map[23][0] + coord_map[24][0]) // 2,
            (coord_map[23][1] + coord_map[24][1]) // 2
        )

    # Mid-ear
    if 7 in coord_map and 8 in coord_map:
        coord_map[999] = (
            (coord_map[7][0] + coord_map[8][0]) // 2,
            (coord_map[7][1] + coord_map[8][1]) // 2
        )
    return coord_map


import numpy as np

def calculate_angle(p1, p2, p3):
    """Calculates the angle at p2 (vertex) using atan2."""
    # Calculate vectors relative to vertex p2
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # Get absolute angles of the vectors from the horizontal
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    # Calculate difference
    diff = angle1 - angle2
    
    # Convert to degrees and ensure the result is positive
    angle = np.abs(np.degrees(diff))
    
    # Ensure we get the interior angle (<= 180)
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def get_angles_error_from_landmarks(coord_map, targets, angle_configs):
    """
    coord_map: Dictionary of {index: (pixel_x, pixel_y)}
    targets: Dict from TOML [pose] section
    angle_configs: List of dicts from TOML angles list
    """
    results = {}
    
    for config in angle_configs:
        name = config["name"]
        point_sets = config["points"]
        
        calculated_angles = []
        for pts in point_sets:
            p1_idx, p2_idx, p3_idx = pts
            
            # Check if all 3 points exist in our pre-calculated coord_map
            if all(idx in coord_map for idx in [p1_idx, p2_idx, p3_idx]):
                ang = calculate_angle(
                    coord_map[p1_idx], 
                    coord_map[p2_idx], 
                    coord_map[p3_idx]
                )
                calculated_angles.append(ang)
        
        if not calculated_angles:
            continue
            
        # Average the angles (useful for multi-point definitions like Spine-Alignment)
        avg_angle = sum(calculated_angles) / len(calculated_angles)
        
        # Calculate error relative to target from TOML
        target_angle = targets.get(name, None)
        error = 0
        if target_angle is not None:
            # error = Actual - Target (e.g., 170 - 180 = -10 deg error)
            error = avg_angle - target_angle
        x = []
        y = []
        for i in range(len(config["points"])):
            x.append(coord_map[config["points"][i][1]][0])
            y.append(coord_map[config["points"][i][1]][1])
        coord = (sum(x) // len(x), sum(y) // len(y))

        

        results[name] = {
            "name": name,
            "coord": coord,
            "current_angle": round(avg_angle, 2),
            "target_angle": target_angle,
            "error": round(error, 2)
        }
    return results


def get_color_gradient(error, threshold):
    """
    Returns a BGR color transitioning from Green (0 error) to Red (at/over threshold).
    """
    severity = min(abs(error) / threshold, 1.0)
    
    red = int(255 * severity)
    green = int(255 * (1.0 - severity))
    blue = 0
    
    return (blue, green, red)


def get_joint_color(idx1, idx2, analysis_results, angle_configs, threshold=20):
    """
    Finds the maximum error from all angles connected to these two landmarks
    and returns the corresponding gradient color.
    """
    max_error = 0
    found = False

    # 1. Find which angles in our results use idx1 or idx2
    # We look at the TOML angle_configs to see the point definitions
    for config in angle_configs:
        name = config["name"]
        if name not in analysis_results:
            continue
            
        # Check if idx1 or idx2 is in any of the point sets for this angle
        # config["points"] looks like [[11, 13, 15], ...]
        for pts in config["points"]:
            if idx1 in pts or idx2 in pts:
                error = abs(analysis_results[name]["error"])
                max_error = max(max_error, error)
                found = True
    
    if not found:
        return (200, 200, 200) # Light Gray if no angle data
        
    return get_color_gradient(max_error, threshold)


def get_lerp_color(current, target, margin):
    """Calculates color from Green (0 err) to Red (margin err)"""
    error = abs(current - target)
    factor = min(error / margin, 1.0)
    r = int(255 * factor)
    g = int(255 * (1 - factor))
    return (0, g, r)


def calculate_joint_feedback(pose, joint, targets, margin, w, h):
    p1 = pose[0]
    p2 = pose[1]
    p3 = pose[2]

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

    return {"angle": angle, "color": (b, g, r), "pos": (int(p2.x * w), int(p2.y * h))}