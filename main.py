import cv2

from src.head_master import HeadMaster
from src.utils import load_toml

def main():
    CONFIG_FILE = "config.toml"
    master = HeadMaster(camera_index=load_toml(CONFIG_FILE)["camera"])

    # Detect key press to start presentation
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Press 's' to start
            master.voice.presentation()
        if key == ord("q"):  # Press 'q' to quit
            master.voice.goodbye()
            master.cleanup()
            break
        if key == ord("d"):  # Press 'd' to demo yoga
            master.do_exercise("chair")
            master.do_exercise("left_warrior2")
            # master.do_exercise("right_warrior2")
            # master.do_exercise("mountain")
        master.process_image()


if __name__ == "__main__":
    main()
