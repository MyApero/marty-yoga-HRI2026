from martypy import Marty
import queue
import random
import threading
import sys
import toml

DEFAULT_ANGLES = {
    "left arm": 0,
    "right arm": 0,
    "left knee": 0,
    "right knee": 0,
    "right twist": 0,
    "left twist": 0,
    "eyes": 0,
    "left hip": 0,
    "right hip": 0,
}


MAPPING = {
    "LeftHip": "left hip",
    "LeftTwist": "left twist",
    "LeftKnee": "left knee",
    "RightHip": "right hip",
    "RightTwist": "right twist",
    "RightKnee": "right knee",
    "LeftArm": "left arm",
    "RightArm": "right arm",
    "Eyes": "eyes",
}


class MyMarty(Marty):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generated_text: callable = None
        self.queue = queue.Queue()
        self.NUMBER_LED = 12
        self.step_length = 75
        threading.Thread(target=self.marty_worker, daemon=True).start()
        if not self.is_conn_ready():
            print("Marty is not connected!")

        self.queue.put((DEFAULT_ANGLES, 1000, False))
        self.is_empty = False

    def init_generated_text(self, generated_text: callable):
        self.generated_text = generated_text

    def get_pose(self):
        all_joints = self.get_joints()

        pose_data = {info["name"]: info["pos"] for info in all_joints.values()}
        return pose_data

    def marty_worker(self):
        while True:
            try:
                type_move, duration, blocking = self.queue.get()
                if isinstance(type_move, dict) and type_move.keys().__contains__("LED"):
                    self.disco_color_eyepicker(
                        colours=type_move["LED"], add_on="LEDeye"
                    )
                elif isinstance(type_move, tuple) and type_move[0].__contains__(
                    "ankle"
                ):
                    joint, angle = type_move
                    if joint.__contains__("right"):
                        self.walk(1, start_foot="right", step_length=self.step_length)
                        self.move_joint("right twist", angle, duration, True)
                        self.move_joint("right twist", 0, duration, True)
                    else:
                        self.walk(1, start_foot="left", step_length=self.step_length)
                        self.move_joint("left twist", angle, duration, True)
                        self.move_joint("left twist", 0, duration, True)
                    self.stand_straight(3 * duration)
                elif isinstance(type_move, tuple) and type_move[0].__contains__("kick"):
                    joint, distance = type_move
                    foot = "right" if joint.__contains__("right") else "left"
                    self.walk(1, start_foot=foot, step_length=distance)
                    self.move_joint("right arm", 0, 1000, True)
                    self.stand_straight(duration)
                else:
                    for joint, angle in type_move.items():
                        self.move_joint(joint, angle, duration, blocking=blocking)
                if self.queue.qsize() == 0:
                    self.is_empty = True
                else:
                    self.is_empty = False
            except Exception as e:
                print(f"Marty Error: {e}", file=sys.stderr)

    def load_pose(self, file: str | dict):
        """
        Translates custom PascalCase pose dict to Marty format and moves.
        """
        translated_pose = {}
        joint_pose = (
            toml.load(file)["marty"] if isinstance(file, str) else file["marty"]
        )

        for custom_key, value in joint_pose.items():
            if custom_key in MAPPING:
                translated_pose[MAPPING[custom_key]] = value

        return translated_pose

    def load_and_do_pose(self, file: str | dict, duration: int = 2000):
        pose = self.load_pose(file)
        for key, value in pose.items():
            self.interaction(key, value, value, False, duration)

    def interaction(
        self, side, height_min, height_max, blocking: bool, duration: int | None = None
    ):
        height = random.randint(height_min, height_max)
        if duration is None:
            duration = abs(height) * 7
        self.queue.put(({side: height}, duration, blocking))
        return duration

    def interaction_eyebrows(self):
        self.queue.put(({"eyes": random.randint(20, 30)}, 100, True))
        self.queue.put(({"eyes": 0}, 150, False))

    def set_light_marty(self, light, time, end_time):
        number_of_light = int(time * self.NUMBER_LED / end_time)
        leds = []
        for _ in range(min(number_of_light + 1, 12)):
            leds.append(light)
        self.queue.put(({"LED": leds}, None, None))

    def move_marty_randomly(self, chunk_duration):
        time_elapsed = 0
        while time_elapsed < chunk_duration:
            wait_time = random.uniform(2, 4)
            max_dice = 5 if time_elapsed == 0 else 3
            dice = random.randint(2, max_dice)
            if dice == 2:
                duration = self.interaction("left arm", 15, 80, True)
                self.interaction("left arm", 0, 0, False, duration)
            if dice == 3:
                duration = self.interaction("right arm", 15, 80, True)
                self.interaction("right arm", 0, 0, False, duration)
            if dice >= 4:
                self.interaction_eyebrows()
            time_elapsed += wait_time
            self.queue.put(({"eyes": 0}, wait_time, True))

    def move_marty_limb(self):
        arm_parts = ["Wrist", "Elbow", "Shoulder"]
        leg_parts = ["Hip", "Knee"]

        text = self.generated_text()
        text_lower = text.lower()

        has_arm = any(part.lower() in text_lower for part in arm_parts)
        has_leg = any(part.lower() in text_lower for part in leg_parts)
        has_spine = "spine" in text_lower
        has_ankle = "ankle" in text_lower

        is_right = "right" in text_lower
        is_left = "left" in text_lower
        if not is_right and not is_left:
            is_left = True
            is_right = True

        if has_arm:
            # Move both if it's a general arm error, or just the specific side
            duration = 0
            if is_right:
                duration = self.interaction("right arm", 100, 100, False)
            if is_left:
                duration = self.interaction("left arm", 100, 100, False)
            if is_right:
                self.interaction("right arm", 0, 0, False, duration)
            if is_left:
                self.interaction("left arm", 0, 0, False, duration)

            if is_right:
                duration = self.interaction("right arm", 100, 100, False)
            if is_left:
                duration = self.interaction("left arm", 100, 100, False)
            if is_right:
                self.interaction("right arm", 0, 0, False, duration)
            if is_left:
                self.interaction("left arm", 0, 0, False, duration)

        # --- BODY (Wiggle vs Kick) ---
        elif has_spine:
            # PRIORITY 1: Wiggle if spine is broken
            duration = self.interaction("right hip", -50, -50, False, 1000)
            self.interaction("left hip", -50, -50, False, duration)
            self.interaction("left hip", -50, -50, False, duration)
            self.interaction("right hip", -50, -50, False, duration)
            self.interaction("right hip", 0, 0, False, duration)
            self.interaction("left hip", 0, 0, False, duration)

        elif has_leg:
            if is_right and not is_left:
                self.queue.put((("right kick", 75), 1500, None))
            elif is_left and not is_right:
                self.queue.put((("left kick", 75), 1500, None))

        elif has_ankle:
            if is_right and not is_left:
                self.queue.put((("right ankle", 50), 500, None))
            if is_left and not is_right:
                self.queue.put((("left ankle", 50), 500, None))
