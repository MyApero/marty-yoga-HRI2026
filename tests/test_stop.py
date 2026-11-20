import threading
import time
from unittest.mock import Mock

import numpy as np

from src.coach_behaviour import CoachBehaviour
from src.speak import Speak


class SlowBody:
    def __init__(self, *args, **kwargs):
        self.interrupted = False

    def is_conn_ready(self):
        return True

    def move_joint(self, *args, **kwargs):
        time.sleep(0.2)

    def interrupt_motion(self):
        self.interrupted = True


class SlowRobot(CoachBehaviour, SlowBody):
    pass


def test_stop_motion_drops_queued_gestures_and_interrupts_the_current_one():
    robot = SlowRobot()
    for _ in range(5):
        robot.interaction("left arm", 40, 40, True, 1000)
    robot.stop_motion()
    assert robot.queue.empty()
    assert robot.interrupted


def test_stop_also_flushes_the_audio_already_buffered():
    voice = Speak.__new__(Speak)
    voice.empty_queues = Mock()
    voice.stream_lock = threading.Lock()
    voice.stream = Mock(active=True)
    voice.subtitles = "Lift your arm"

    voice.stop()

    voice.empty_queues.assert_called_once()
    voice.stream.abort.assert_called_once()
    voice.stream.start.assert_called_once()
    assert voice.subtitles == ""


def test_a_sentence_waiting_for_its_turn_is_dropped_after_stop():
    voice = Speak.__new__(Speak)
    voice.shutdown_event = threading.Event()
    voice.stream_lock = threading.Lock()
    voice.stream = Mock(active=True)
    voice.current_epoch = 0
    voice.move_marty_enabled = True
    voice.move_marty_callback = Mock()
    voice.move_marty_type_corrective = None
    voice.show_voice = None
    allowed = threading.Event()
    voice.can_i_speak = allowed.is_set

    player = threading.Thread(target=voice._play_chunk_wrapper, args=(np.zeros(24000, np.float32), 0))
    player.start()
    time.sleep(0.15)
    voice.current_epoch = 1
    allowed.set()
    player.join(timeout=2)

    voice.stream.write.assert_not_called()
    voice.move_marty_callback.assert_not_called()
