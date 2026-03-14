class FeedbackEngine:
    def __init__(self, max_error_margin, send_correction_threshold=0.7):
        self.max_error_margin = max_error_margin
        self.send_correction_threshold = send_correction_threshold
        self.ongoing_mistakes = {}

    def reset(self):
        self.ongoing_mistakes.clear()

    def update_ongoing_frame(self, actual_run, elapsed):
        if len(actual_run) == 0:
            return

        frame = actual_run[-1]
        for angle_name, angle_data in frame.items():
            if (
                abs(angle_data["error"])
                < self.max_error_margin * self.send_correction_threshold
            ):
                if (
                    angle_name in self.ongoing_mistakes
                    and len(self.ongoing_mistakes[angle_name]["mistakes"][-1]) < 2
                ):
                    self.ongoing_mistakes[angle_name]["mistakes"][-1].append(elapsed)
                    self.ongoing_mistakes[angle_name]["timed_mistake"] += (
                        self.ongoing_mistakes[angle_name]["mistakes"][-1][1]
                        - self.ongoing_mistakes[angle_name]["mistakes"][-1][0]
                    )
                continue

            if angle_name not in self.ongoing_mistakes:
                self.ongoing_mistakes[angle_name] = {
                    "mistakes_repetitions": 0,
                    "timed_mistake": 0,
                    "remider_done": 0,
                    "target_angle": angle_data["target_angle"],
                    "current_angle": angle_data["current_angle"],
                    "mistakes": [[elapsed]],
                }

            if len(self.ongoing_mistakes[angle_name]["mistakes"][-1]) > 1:
                self.ongoing_mistakes[angle_name]["mistakes"].append([elapsed])
                self.ongoing_mistakes[angle_name]["mistakes_repetitions"] += 1

    def analyze_ongoing_frame(self):
        correction_to_do = {}
        for angle_name, mistakes in self.ongoing_mistakes.items():
            if len(mistakes["mistakes"][-1]) < 2:
                correction_to_do[
                    angle_name + " target:" + str(round(mistakes["target_angle"]))
                ] = "current:" + str(round(mistakes["current_angle"]))
                mistakes["remider_done"] += 1
        return correction_to_do
