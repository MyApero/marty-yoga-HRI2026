import numpy as np

from src.voice_leds import BAND_EDGES_HZ, FPS, band_levels, eye_frame

RATE = 24000


def tone(hz, seconds=1.0):
    return np.sin(2 * np.pi * hz * np.arange(int(RATE * seconds)) / RATE).astype(np.float32)


def test_silence_keeps_the_eyes_dark():
    assert not band_levels(np.zeros(RATE, np.float32), RATE).any()


def test_a_tone_lights_the_led_of_its_band():
    hz = 1000
    expected = int(np.digitize(hz, BAND_EDGES_HZ)) - 1
    levels = band_levels(tone(hz), RATE)
    assert levels.shape == (FPS, 12)
    assert (levels[5:-1].argmax(axis=1) == expected).all()  # the last window runs past the end of the tone


def test_leds_fade_out_after_the_sound_stops():
    levels = band_levels(np.concatenate([tone(500, 0.5), np.zeros(RATE // 2, np.float32)]), RATE)
    loudness = levels.max(axis=1)
    assert loudness[16] > loudness[20] > loudness[-1]


def test_right_eye_mirrors_the_left():
    frame = eye_frame(np.linspace(0, 1, 12))
    assert len(frame) == 24
    assert frame[:12] == frame[12:][::-1]
    assert frame[0] == 0 and frame[11] == 0x00C8FF
