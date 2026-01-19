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
}


class MyMarty(Marty):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.queue = queue.Queue()
        threading.Thread(target=self.marty_worker, daemon=True).start()
        if not self.is_conn_ready():
            print("Marty is not connected!")

        self.queue.put((DEFAULT_ANGLES, 100, False))

    def get_pose(self):
        all_joints = self.get_joints()

        pose_data = {info['name']: info['pos'] for info in all_joints.values()}
        return pose_data

    def marty_worker(self):
        while True:
            try:
                positions, duration, blocking = self.queue.get()
                for joint, angle in positions.items():
                    self.move_joint(joint, angle, duration, blocking=blocking)
            except Exception as e:
                print(f"Marty Error: {e}", file=sys.stderr)

    def load_pose(self, file: str|dict):
        """
        Translates custom PascalCase pose dict to Marty format and moves.
        """
        translated_pose = {}
        print(file)
        joint_pose = toml.load(file)["marty"] if isinstance(file, str) else file["marty"]

        mapping = {
        "LeftHip": "left hip",
        "LeftTwist": "left twist",
        "LeftKnee": "left knee",
        "RightHip": "right hip",
        "RightTwist": "right twist",
        "RightKnee": "right knee",
        "LeftArm": "left arm",
        "RightArm": "right arm",
        "Eyes": "eyes"
        }

        for custom_key, value in joint_pose.items():
            if custom_key in mapping:
                translated_pose[mapping[custom_key]] = value
        
        return translated_pose
    


    def load_and_do_pose(self, file:str|dict, duration:int=1000):
        pose = self.load_pose(file)
        for key, value in pose.items():
            self.interaction(key, value, value, False, duration)


    def interaction(self, side, height_min, height_max, bloking:bool, duration:int|None=None):
        arm_height = random.randint(height_min, height_max)
        if duration is None:
            duration = arm_height * 7
        self.queue.put(({side: arm_height}, duration, bloking))
        return duration

    def interaction_eyebrows(self):
        self.queue.put(({"eyes": random.randint(20, 30)}, 100, True))
        self.queue.put(({"eyes": 0}, 150, False))

    def move_marty_arm_randomly(self, chunk_duration):
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
