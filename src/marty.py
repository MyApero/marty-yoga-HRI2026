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
        threading.Thread(target=self.marty_worker, daemon=True).start()
        if not self.is_conn_ready():
            print("Marty is not connected!")

        self.queue.put((DEFAULT_ANGLES, 1000, False))

    def init_generated_text(self, generated_text: callable):
        self.generated_text = generated_text

    def get_pose(self):
        all_joints = self.get_joints()

        pose_data = {info["name"]: info["pos"] for info in all_joints.values()}
        return pose_data

    def marty_worker(self):
        while True:
            try:
                positions, duration, blocking = self.queue.get()
                for joint, angle in positions.items():
                    self.move_joint(joint, angle, duration, blocking=blocking)
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
        self, side, height_min, height_max, bloking: bool, duration: int | None = None
    ):
        arm_height = random.randint(height_min, height_max)
        if duration is None:
            duration = arm_height * 7
        self.queue.put(({side: arm_height}, duration, bloking))
        return duration

    def interaction_eyebrows(self):
        self.queue.put(({"eyes": random.randint(20, 30)}, 100, True))
        self.queue.put(({"eyes": 0}, 150, False))

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
        leg_parts = ["Hip", "Knee", "Ankle"]

        text = self.generated_text()
        limb_lower = text.lower()

        has_arm = any(part.lower() in limb_lower for part in arm_parts)
        has_leg = any(part.lower() in limb_lower for part in leg_parts)
        has_spine = "spine" in limb_lower

        is_right = "right" in limb_lower
        is_left = "left" in limb_lower

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
        # if has_spine:
        #     # PRIORITY 1: Wiggle if spine is broken
        #     print("Spine issue detected: Wiggling...")
        #     duration = self.interaction("right hip", 100, 100, False)
        #     self.interaction("left hip", -100, 100, False, duration)
        #     self.interaction("right hip", 0, 0, False, duration)
        #     self.interaction("left hip", 0, 0, False, duration)

        elif has_leg:
            if is_right and not is_left:
                self.kick("right", blocking=True)
            elif is_left and not is_right:
                self.kick("left")
