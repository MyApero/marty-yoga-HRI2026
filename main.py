import cv2

from src.head_master import HeadMaster

def main():
    master = HeadMaster(camera_index=2)

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
            master.load_pose("mountain")
            master.do_pose()
        master.process_image()


if __name__ == "__main__":
    main()
