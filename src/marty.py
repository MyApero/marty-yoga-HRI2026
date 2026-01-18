from martypy import Marty
import queue
import random
import threading
import sys

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

############ A TESTER ###############
    def interraction_knee(self, side, height_min, height_max):
        leg_height = random.randint(height_min, height_max)
        duration = leg_height * 7
        self.queue.put(({side: leg_height}, duration, True))
        self.queue.put(({side: 0}, duration, False))

    def interraction_ankel_rot(self, side, height_min, height_max):
        ankel_rot = random.randint(height_min, height_max)
        duration = ankel_rot * 7
        self.queue.put(({side: ankel_rot}, duration, True))
        self.queue.put(({side: 0}, duration, False))

    def interraction_ankel_dev(self, side, height_min, height_max):
        ankel_dev = random.randint(height_min, height_max)
        duration = ankel_dev * 7
        self.queue.put(({side: ankel_dev}, duration, True))
        self.queue.put(({side: 0}, duration, False))
####################################

    def interaction_arm(self, side, height_min, height_max):
        arm_height = random.randint(height_min, height_max)
        duration = arm_height * 7
        self.queue.put(({side: arm_height}, duration, True))
        self.queue.put(({side: 0}, duration, False))

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
                self.interaction_arm("left arm", 15, 80)
            if dice == 3:
                self.interaction_arm("right arm", 15, 80)
            if dice >= 4:
                self.interaction_eyebrows()
            time_elapsed += wait_time
            self.queue.put(({"eyes": 0}, wait_time, True))
