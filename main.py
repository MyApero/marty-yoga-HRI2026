import cv2
from dataclasses import dataclass

from src.head_master import HeadMaster


@dataclass
class Pose:
    name: str
    mediapipe_pose: str
    be_careful_at: str


def main():
    master = HeadMaster(camera_index=2)

    # Detect key press to start presentation
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Press 's' to start
            master.marty.presentation()
        if key == ord("q"):  # Press 'q' to quit
            master.marty.goodbye()
            master.cleanup()
            break
        if key == ord("d"):  # Press 'd' to demo yoga
            master.load_pose("warrior")
            master.do_pose()
        master.process_image()


if __name__ == "__main__":
    main()


# say_bonjour()

# for pose in poses:
#     print(f"Now demonstrating the {pose.name} pose.")
#     headmaster.load_pose(pose)
#     headmaster.do_pose()
