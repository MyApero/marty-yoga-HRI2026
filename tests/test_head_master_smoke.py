import unittest
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock


def _install_test_stubs():
    cv2_stub = ModuleType("cv2")
    cv2_stub.VideoCapture = Mock
    cv2_stub.waitKey = lambda *_args, **_kwargs: -1
    cv2_stub.destroyAllWindows = Mock()
    cv2_stub.imshow = Mock()
    sys.modules.setdefault("cv2", cv2_stub)

    mp_stub = ModuleType("mediapipe")
    mp_stub.Image = Mock
    mp_stub.ImageFormat = SimpleNamespace(SRGB=0)
    sys.modules.setdefault("mediapipe", mp_stub)

    mediapipe_ops_stub = ModuleType("src.mediapipe_operations")
    mediapipe_ops_stub.setup_landmarker = Mock(return_value=Mock())
    mediapipe_ops_stub.apply_film_effect = Mock(side_effect=lambda img, _cfg: img)
    sys.modules.setdefault("src.mediapipe_operations", mediapipe_ops_stub)

    marty_stub = ModuleType("src.marty")
    marty_stub.MyMarty = Mock
    sys.modules.setdefault("src.marty", marty_stub)

    speak_stub = ModuleType("src.speak")
    speak_stub.Speak = Mock
    sys.modules.setdefault("src.speak", speak_stub)

    camera_stub = ModuleType("src.camera")
    camera_stub.capture_image_from_camera = Mock()
    sys.modules.setdefault("src.camera", camera_stub)

    window_stub = ModuleType("src.window")
    window_stub.WindowRenderer = Mock
    sys.modules.setdefault("src.window", window_stub)


_install_test_stubs()

from src.head_master import HeadMaster


class TestHeadMasterSmoke(unittest.TestCase):
    def setUp(self):
        self.master = HeadMaster.__new__(HeadMaster)
        self.master.voice = SimpleNamespace(intro=Mock())
        self.master.logger = SimpleNamespace(setLevel=Mock())
        self.master.cleanup = Mock()
        self.master.generate_yoga_images_with_landmarks = Mock()
        self.master.run_demo = Mock()
        self.master.process_camera_image = Mock()

    def test_tick_dispatches_start_and_renders_frame(self):
        keep_running = self.master.tick(ord("s"))

        self.assertTrue(keep_running)
        self.master.voice.intro.assert_called_once()
        self.master.process_camera_image.assert_called_once()

    def test_tick_stops_on_quit(self):
        keep_running = self.master.tick(ord("q"))

        self.assertFalse(keep_running)
        self.master.cleanup.assert_called_once()
        self.master.process_camera_image.assert_not_called()

    def test_tick_dispatches_demo(self):
        keep_running = self.master.tick(ord("d"))

        self.assertTrue(keep_running)
        self.master.run_demo.assert_called_once()
        self.master.process_camera_image.assert_called_once()


if __name__ == "__main__":
    unittest.main()
