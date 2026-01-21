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

MAPPING_MEDIAPIPE_MARTY = {
    "Right Wrist" : "Right Arm",
    "Left Wrist" : "Left Arm",
    "Right Elbow" : "Right Arm",
    "Left Elbow" : "Left Arm",
    "Right Shoulder" : "Arms",
    "Left Shoulder" : "Arms",
    "Right Knee" : "Right Knee",
    "Left Knee" : "Left Knee",
    "Right Hip" : "Hips",
    "Left Hip" : "Hips",
    "Spine" : "Spine",
    "Spine Alignment" : "Spine",
    "Right Ankle" : "Right Twist",
    "Left Ankle" : "Left Twist",
}

class MyMarty(Marty):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.queue = queue.Queue()
        threading.Thread(target=self.marty_worker, daemon=True).start()
        if not self.is_conn_ready():
            print("Marty is not connected!")

        self.queue.put((DEFAULT_ANGLES, 1000, False))

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


    def move_marty_limb(self, chunk_duration, limb):
        time_elapsed = 0

        arm_tracker = ["Wrist", "Elbow", "Shoulder"]
        leg_tracker = ["Hip", "Knee", "Ankle"]
        spine_tracker = ["Spine"]
        
        limb_keys = list(limb.keys())

        while time_elapsed < chunk_duration:
            has_arm = any(t in k for k in limb_keys for t in arm_tracker)
            has_leg = any(t in k for k in limb_keys for t in leg_tracker)
            has_spine = any(t in k for k in limb_keys for t in spine_tracker)
            
            has_right = any("Right" in k for k in limb_keys)
            has_left = any("Left" in k for k in limb_keys)

            if has_arm:
                if has_right:
                    self.interaction("right arm", 100, 100, False)
                if has_left:
                    self.interaction("left arm", 100, 100, False)
                if has_right:
                    self.interaction("right arm", 0, 0, False)
                if has_left:
                    self.interaction("left arm", 0, 0, False)

            if has_spine:
                duration = self.interaction("right hip", 100, 100, True)
                self.interaction("left hip", -100, 100, True, duration)
                # Reset
                self.interaction("right hip", 0, 0, True, duration)
                self.interaction("left hip", 0, 0, True, duration)
                
            elif has_leg:
                if has_right and not has_left:
                    self.kick("right")
                elif has_left and not has_right:
                    self.kick("left")

            wait_time = random.uniform(6, 8)
            time_elapsed += wait_time


