def variance_feedback(actual_run, feedback_summary):
    """
    Analyzes the variance in joint angles over the actual_run images
    and provides feedback on consistency.
    """

    angles = {}
    for frame in actual_run:
        for angle_name, angle_data in frame.items():
            if angle_name not in angles:
                angles[angle_name] = []
            angles[angle_name].append(angle_data["current_angle"])

    mean_angles = {name: sum(values) / len(values) for name, values in angles.items()}
    for name, values in angles.items():
        variance = sum((x - mean_angles[name]) ** 2 for x in values) / len(values)
        feedback_summary[name]["Variance"] = "Good" if variance < 15 else "Needs Improvement"
        # {
        #     "mean_angle": round(mean_angles[name], 2),
        #     "variance": round(variance, 2),
        #     "consistency": "Good" if variance < 15 else "Needs Improvement",
        # }

def get_pose_validity(actual_run, feedback_summary):
    angles = {}
    for frame in actual_run:
        for angle_name, angle_data in frame.items():
            if angle_name not in angles:
                angles[angle_name] = {"angle": [], "target": angle_data["target_angle"]}
            angles[angle_name]["angle"].append(angle_data["current_angle"])

    for angle_name in angles:
        angles[angle_name]["minimum"] = min(angles[angle_name]["angle"])
        angles[angle_name]["maximum"] = max(angles[angle_name]["angle"])
        angles[angle_name]["mean"] = sum(angles[angle_name]["angle"]) / len(
            angles[angle_name]["angle"]
        )
        del angles[angle_name]["angle"]

    # difference between mean_angle and target_angle
    for angle_name in angles:
        error = round(
            abs(angles[angle_name]["mean"] - angles[angle_name]["target"]), 2
        )
        feedback_summary[angle_name]["Validity"] = "Valid" if error < 10 else "Invalid"


def get_errors_over_time(actual_run, time, max_error, feedback_summary):
    error_timeline = {}
    frame_number = len(actual_run)
    for frame in actual_run:
        for angle_name, angle_data in frame.items():
            if angle_data["error"] is None or angle_data["error"] < max_error:
                continue
            if angle_name not in error_timeline:
                error_timeline[angle_name] = 0
            error_timeline[angle_name] += 1

    for angle_name, error in error_timeline.items():
        error_timeline[angle_name] = round(float(time * error / frame_number), 1)

    # Error time of each angle in seconds
    for angle_name, total_error_time in error_timeline.items():
        feedback_summary[angle_name]["Total error time sec"] = str(total_error_time) + "s"

def get_feedbacks_from_run(actual_run, time, max_error):
    """
    actual_run: List of images with drawn skeletons from the pose attempt
    pose: Dict from TOML [pose] section
    history: List to append feedback summaries to
    """
    feedback_summary = {}
    # Fill that with every joints
    for frame in actual_run:
        for angle_name, _ in frame.items():
            if angle_name not in feedback_summary:
                feedback_summary[angle_name] = {
                    "Variance": '',
                    "Validity": '',
                }

    variance_feedback(actual_run, feedback_summary)
    get_pose_validity(actual_run, feedback_summary)
    get_errors_over_time(actual_run, time, max_error, feedback_summary)

    # feedback_summary = {
    #     "variance_feedback": variance,  # Feedback on angle consistency over time
    #     "pose_validity": validity,  # Summary of angle accuracy against targets
    #     "errors_over_time": errors_over_time,
    #     # à vérifier !
    # }

    return feedback_summary
