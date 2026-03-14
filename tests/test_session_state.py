import unittest

from src.session_state import SessionState


class TestSessionState(unittest.TestCase):
    def test_defaults(self):
        state = SessionState()

        self.assertIsNone(state.pose_name)
        self.assertEqual(state.actual_run, [])
        self.assertEqual(state.history, [])
        self.assertIsNone(state.name_files)
        self.assertTrue(state.pose_ended)
        self.assertFalse(state.is_pose_ending)

    def test_list_fields_are_isolated_per_instance(self):
        a = SessionState()
        b = SessionState()

        a.actual_run.append({"frame": 1})
        a.history.append("entry")

        self.assertEqual(a.actual_run, [{"frame": 1}])
        self.assertEqual(a.history, ["entry"])
        self.assertEqual(b.actual_run, [])
        self.assertEqual(b.history, [])


if __name__ == "__main__":
    unittest.main()
