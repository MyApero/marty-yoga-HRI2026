import logging
from src.utils import loading_print
from src.app_cli import (
    configure_startup_logging,
    get_current_config,
    normalize_gen_poses,
    parse_args,
    silence_native_output,
)

configure_startup_logging()

loading_print("[1/4] Loading core libraries...")
import cv2

loading_print("[2/4] OpenCV loaded.")

loading_print("[3/4] Loading AI models (mediapipe + kokoro, please wait)...")
from src.head_master import HeadMaster

loading_print("[4/4] All models loaded. Starting Yoga Master...")


def run_generation(master, gen_poses):
    if gen_poses:
        saved_files = master.generate_yoga_images_with_landmarks(
            poses=gen_poses,
            verbose=False,
        )
    else:
        saved_files = master.generate_yoga_images_with_landmarks(
            verbose=False,
        )

    for output_filename in saved_files:
        print(f"Saved {output_filename}")


def run_interactive(master):
    while True:
        key = cv2.waitKey(1) & 0xFF
        if not master.tick(key):
            break


def main():
    print("\n")
    args = parse_args()
    current_config = get_current_config()
    master = None

    try:
        gen_poses = normalize_gen_poses(args.gen)
        if gen_poses is not None:
            with silence_native_output():
                master = HeadMaster(
                    current_config=current_config,
                    logging_level=logging.WARNING,
                    enable_marty=False,
                    enable_voice=False,
                )
            run_generation(master, gen_poses)
            return

        master = HeadMaster(current_config=current_config)
        run_interactive(master)
    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")
    finally:
        if master is not None:
            master.cleanup()


if __name__ == "__main__":
    main()
