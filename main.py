import platform
import logging
from src.utils import loading_print, load_toml

loading_print("[1/4] Loading core libraries...")
import cv2
loading_print("[2/4] OpenCV loaded.")

loading_print("[3/4] Loading AI models (mediapipe + kokoro, please wait)...")
from src.head_master import HeadMaster
loading_print("[4/4] All models loaded. Starting Yoga Master...")


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
    print("\n")
    master = HeadMaster(current_config=current_config)

    # Detect key press to start presentation
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Press 's' to start
            master.voice.intro()
        if key == ord("q"):  # Press 'q' to quit
            master.cleanup()
            break
        if key == ord("y"):  # Press 'y' to generate yoga images with landmarks
            master.logger.setLevel(logging.WARNING)
            master.generate_yoga_images_with_landmarks()
        if key == ord("d"):  # Press 'd' to demo yoga
            master.voice.intro()
            master.do_exercise("chair")
            master.do_exercise("left_warrior2")
            # master.do_exercise("right_warrior2")
            # master.do_exercise("mountain")
        master.process_camera_image()


if __name__ == "__main__":
    main()
