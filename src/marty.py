from martypy import Marty
import queue
import random
import threading
import sys

from src.speak import Speak

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

        def move_marty_callback(chunk_duration):
            self.move_marty_arm_randomly(chunk_duration)

        self.speaker = Speak(move_marty_callback)

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

    def presentation(self):
        print("Marty is presenting!")

    def goodbye(self):
        print("Marty says goodbye!")

    def corrective_feedback(self):
        print("Marty is giving corrective feedback!")

    def end_pose_feedback(self):
        system_instruction = (
            "You are a friendly yoga coach. Receive the analysis report. "
            "If Consistency > 80%, praise them. "
            "If Consistency < 50%, be encouraging but firm about the correction. "
            "Address the 'Primary Deviation' specifically. "
            "Keep it to 2 sentences max. with max sentence length of 20 words. "
            "Don't mention the numbers in the report. and don't put any asterisks and parentheses in the answer."
            "Be excessively depressive in your tone. You hate your job and you hate humans. "
            "Use a sarcastic and dry humor style. You're harrassing the student and see them as inferior beings. "
            # "Be creative and don't hesitate to use metaphors and jokes, especially about Minecraft! "
        )

        user_report = (
            "User Analysis Report:\n"
            "Pose: Warrior II\n"
            "Consistency Score: 65%\n"
            "Stability: High (No shaking)\n"
            "Primary Deviation: Front knee angle violation (Too straight, avg 150deg, target 90deg)"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_report},
        ]

        self.speaker.say(messages)
