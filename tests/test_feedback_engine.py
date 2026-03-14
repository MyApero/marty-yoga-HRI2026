import unittest

from src.feedback_engine import FeedbackEngine


class TestFeedbackEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FeedbackEngine(
            max_error_margin=10.0, send_correction_threshold=0.7
        )

    def _frame(self, error, target=90.0, current=70.2):
        return {
            "Right Knee": {
                "error": error,
                "target_angle": target,
                "current_angle": current,
            }
        }

    def test_analyze_returns_correction_for_open_mistake(self):
        actual_run = [self._frame(error=9.0)]

        self.engine.update_ongoing_frame(actual_run, elapsed=1.0)
        correction = self.engine.analyze_ongoing_frame()

        self.assertIn("Right Knee target:90", correction)
        self.assertEqual(correction["Right Knee target:90"], "current:70")
        self.assertEqual(
            self.engine.ongoing_mistakes["Right Knee"]["reminder_done"],
            1,
        )

    def test_closed_interval_is_not_reported_as_correction(self):
        actual_run = [self._frame(error=9.0)]
        self.engine.update_ongoing_frame(actual_run, elapsed=2.0)

        actual_run = [self._frame(error=2.0)]
        self.engine.update_ongoing_frame(actual_run, elapsed=4.0)

        correction = self.engine.analyze_ongoing_frame()

        self.assertEqual(correction, {})
        self.assertEqual(
            self.engine.ongoing_mistakes["Right Knee"]["timed_mistake"],
            2.0,
        )

    def test_repetition_counter_increments_on_new_mistake_window(self):
        self.engine.update_ongoing_frame([self._frame(error=9.0)], elapsed=1.0)
        self.engine.update_ongoing_frame([self._frame(error=1.0)], elapsed=2.0)
        self.engine.update_ongoing_frame([self._frame(error=9.0)], elapsed=3.0)

        angle = self.engine.ongoing_mistakes["Right Knee"]
        self.assertEqual(angle["mistakes_repetitions"], 1)
        self.assertEqual(len(angle["mistakes"]), 2)
        self.assertEqual(angle["mistakes"][1], [3.0])

    def test_reset_clears_ongoing_mistakes(self):
        self.engine.update_ongoing_frame([self._frame(error=9.0)], elapsed=1.0)
        self.assertTrue(self.engine.ongoing_mistakes)

        self.engine.reset()

        self.assertEqual(self.engine.ongoing_mistakes, {})


if __name__ == "__main__":
    unittest.main()
