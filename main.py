import cv2

from src.head_master import HeadMaster
from src.utils import load_toml

import platform

CONFIG_FILE = "config.toml"
current_os = platform.system().lower()
config = load_toml(CONFIG_FILE)

if current_os == "darwin":
    current_config = config["config_macos"]
elif current_os == "linux":
    current_config = config["config_linux"]
else:
    current_config = config["config_common"]


def main():
    master = HeadMaster(current_config=current_config)

    # Detect key press to start presentation
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Press 's' to start
            master.voice.intro()
        if key == ord("q"):  # Press 'q' to quit
            master.cleanup()
            break
        if key == ord("d"):  # Press 'd' to demo yoga
            master.voice.intro()
            master.do_exercise("chair")
            master.do_exercise("left_warrior2")
            # master.do_exercise("right_warrior2")
            # master.do_exercise("mountain")
        master.process_image()


if __name__ == "__main__":
    main()
