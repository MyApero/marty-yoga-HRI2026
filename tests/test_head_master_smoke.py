import queue
import threading
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

    live_pose_stub = ModuleType("src.live_pose")
    live_pose_stub.LivePose = Mock
    sys.modules.setdefault("src.live_pose", live_pose_stub)

    window_stub = ModuleType("src.window")
    window_stub.WindowRenderer = Mock
    sys.modules.setdefault("src.window", window_stub)


_STUBBED = {"cv2", "mediapipe", "src.mediapipe_operations", "src.marty", "src.speak", "src.camera", "src.window", "src.live_pose"}
_already_loaded = _STUBBED & set(sys.modules)
_install_test_stubs()

from src.head_master import STOP_SETTLE_S, HeadMaster, SessionStopped

# Stubs only while head_master imports, so the other test files get the real modules.
for _name in _STUBBED - _already_loaded:
    sys.modules.pop(_name, None)


class TestHeadMasterSmoke(unittest.TestCase):
    def setUp(self):
        self.master = HeadMaster.__new__(HeadMaster)
        self.master.voice = SimpleNamespace(intro=Mock())
        self.master.logger = SimpleNamespace(setLevel=Mock())
        self.master.cleanup = Mock()
        self.master.generate_yoga_images_with_landmarks = Mock()
        self.master.run_demo = Mock()
        self.master.process_camera_image = Mock()
        self.master.stop_requested = threading.Event()
        self.master.plugin_commands = queue.Queue()
        self.master.session_label = None
        self.master.stopping = False
        self.master.marty = None
        self.master.starting = False
        self.master.plugins = []
        self.master.ignore_escape_until = 0
        self.master.stop_session = Mock()

    def test_tick_dispatches_start_and_renders_frame(self):
        keep_running = self.master.tick(ord("s"))

        self.assertTrue(keep_running)
        self.master.voice.intro.assert_called_once()
        self.master.process_camera_image.assert_called_once()

    def test_tick_stops_on_quit(self):
        keep_running = self.master.tick(27)  # Esc; "q" is the coach-Chair key

        self.assertFalse(keep_running)
        self.master.cleanup.assert_called_once()
        self.master.process_camera_image.assert_not_called()

    def test_tick_dispatches_demo(self):
        keep_running = self.master.tick(ord("d"))

        self.assertTrue(keep_running)
        self.master.run_demo.assert_called_once()
        self.master.process_camera_image.assert_called_once()

    def test_lucy_stop_ends_a_running_session(self):
        def coach_until_stopped(_pose):
            self.master.on_plugin_action("stop", None)
            self.master.update_window()
            raise AssertionError("the session kept running after Stop")

        self.master.do_mediapipe_pose_only = coach_until_stopped
        self.master.start("coach", "chair")

        self.master.stop_session.assert_called_once()
        self.assertIsNone(self.master.session_label)
        self.assertFalse(self.master.stop_requested.is_set())

    def test_lucy_commands_run_on_the_main_loop(self):
        self.master.do_marty_pose_only = Mock()
        self.master.on_plugin_action("demonstrate", "mountain")
        self.master.do_marty_pose_only.assert_not_called()

        self.master.tick(255)  # no key pressed

        self.master.do_marty_pose_only.assert_called_once_with("mountain")

    def test_stop_while_idle_does_not_stop_the_next_session(self):
        self.master.on_plugin_action("stop", None)
        self.assertFalse(self.master.stop_requested.is_set())

    def test_unknown_requests_start_nothing(self):
        self.assertIsNone(self.master.session_for("coach", "not_a_pose"))
        self.assertIsNone(self.master.session_for("full_session", "99"))
        self.assertIsNone(self.master.session_for("dance"))

    def test_demo_button_plays_when_ready_and_stops_while_running(self):
        ready = self.master.demo_manifest()
        self.assertEqual((ready["caption"], [a["id"] for a in ready["actions"]]), ("DEMO", ["demo"]))
        self.master.session_label = "Demo"
        running = self.master.demo_manifest()
        self.assertEqual((running["caption"], [a["id"] for a in running["actions"]]), ("STOP", ["stop"]))
        self.assertTrue(running["active"])

    def test_yoga_menu_offers_stop_only_while_busy(self):
        def enabled(manifest):
            return {a["id"] for a in manifest["actions"] if not a.get("disabled")}

        self.assertEqual(enabled(self.master.plugin_manifest()), {"full_session", "coach", "demonstrate", "intro"})
        self.master.session_label = "Coaching Chair"
        self.assertEqual(enabled(self.master.plugin_manifest()), {"stop"})

    def test_lucy_demo_runs_what_the_d_key_runs(self):
        self.master.on_plugin_action("demo", None)
        self.master.tick(255)

        self.master.run_demo.assert_called_once()

    def test_stop_spins_until_speech_motion_and_rest_pose_are_done(self):
        captions = []
        master = self.master
        master.plugins = [SimpleNamespace(announce=lambda: captions.append(master.demo_manifest()["caption"]))]
        master.session_label = "Demo"
        master.marty = Mock()
        master.voice = Mock()
        master.session = SimpleNamespace(pose_ended=False, is_pose_ending=True)
        master.set_interaction_state = Mock()
        master.settle = Mock()

        HeadMaster.stop_session(master)

        self.assertEqual(captions, ["STOPPING"])
        master.voice.stop.assert_called_once()
        master.marty.stop_motion.assert_called_once()
        master.marty.disco_off.assert_called_once()
        master.settle.assert_called_once_with(STOP_SETTLE_S)
        self.assertTrue(master.session.pose_ended)

    def test_a_second_stop_while_stopping_is_ignored(self):
        self.master.session_label = "Demo"
        self.master.stopping = True
        self.master.on_plugin_action("stop", None)
        self.assertFalse(self.master.stop_requested.is_set())

    def test_demo_spins_while_starting_and_can_still_be_stopped(self):
        captions = []
        master = self.master
        master.plugins = [SimpleNamespace(announce=lambda: captions.append(master.demo_manifest()))]
        master.run_demo = lambda poses: master.on_voice_played()

        master.start("demo")

        starting, playing = captions[0], captions[1]
        self.assertEqual((starting["icon"], starting["caption"]), ("loading", "STOP"))
        self.assertEqual(starting["actions"], [{"id": "stop", "label": "Stop", "danger": True}])
        self.assertEqual(playing["icon"], "stop")
        self.assertFalse(master.starting)

    def test_sessions_that_open_silently_do_not_spin(self):
        self.master.do_mediapipe_pose_only = lambda pose: None
        self.master.plugins = [SimpleNamespace(announce=lambda: None)]
        self.master.session_for = lambda action, option=None: ("Coaching Chair", lambda: self.assertFalse(self.master.starting))
        self.master.start("coach", "chair")

    def test_a_double_click_on_demo_starts_one_demo(self):
        self.master.on_plugin_action("demo", None)
        self.master.on_plugin_action("demo", None)
        self.assertEqual(self.master.plugin_commands.qsize(), 1)

    def test_start_requests_during_a_session_are_dropped(self):
        self.master.session_label = "Demo"
        self.master.on_plugin_action("demo", None)
        self.master.on_plugin_action("coach", "chair")
        self.assertTrue(self.master.plugin_commands.empty())

    def test_the_reachy_demo_picks_mountain_or_warrior_2(self):
        self.master.marty = SimpleNamespace(robot_package="reachy_mini_so101_humanoid_urdf")
        self.assertEqual(self.master.demo_pool(), ["mountain", "left_warrior2"])

    def test_the_g1_demo_runs_warrior_2_and_offers_nothing_else(self):
        self.master.marty = SimpleNamespace(robot_package="unitree_g1_urdf")
        self.master.on_plugin_action("demo", None)
        self.master.tick(255)

        self.master.run_demo.assert_called_once_with(poses=["right_warrior2"])
        self.assertIsNone(self.master.session_for("coach", "chair"))
        coach = next(a for a in self.master.plugin_manifest()["actions"] if a["id"] == "coach")
        self.assertEqual([o["id"] for o in coach["options"]], ["right_warrior2"])

    def test_other_robots_keep_every_pose(self):
        self.master.marty = SimpleNamespace(robot_package="marty_urdf")
        self.assertEqual(len(self.master.supported_poses()), 7)

if __name__ == "__main__":
    unittest.main()
