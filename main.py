import platform
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

    while True:
        key = cv2.waitKey(1) & 0xFF
        if not master.tick(key):
            break


if __name__ == "__main__":
    main()
