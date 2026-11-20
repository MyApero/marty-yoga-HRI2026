"""Turn speech audio into eye LED frames: each LED of a ring shows how loud one frequency band is."""

import numpy as np

LEDS_PER_EYE = 12
FPS = 30
WINDOW = 1024
BAND_EDGES_HZ = np.geomspace(90, 6000, LEDS_PER_EYE + 1)  # voice range, one band per LED, low to high
COLOUR = 0x00C8FF
DECAY = 0.65  # per frame; LEDs fade instead of blinking off between syllables
GAMMA = 2.2  # LEDs look linear in brightness only after this curve
QUIET_BAND = 0.05  # a band never louder than 5% of the voice's peak stays dark instead of being amplified


def band_levels(audio, sample_rate: int, fps: int = FPS) -> np.ndarray:
    """One row per LED frame, one column per band, 0-1 relative to that band's loudness over the chunk."""
    audio = np.asarray(audio, dtype=np.float32).ravel()
    hop = sample_rate / fps
    count = max(1, int(len(audio) / hop))
    starts = (np.arange(count) * hop).astype(int)
    padded = np.pad(audio, (0, WINDOW))
    spectrum = np.abs(np.fft.rfft(padded[starts[:, None] + np.arange(WINDOW)] * np.hanning(WINDOW), axis=1))
    band = np.digitize(np.fft.rfftfreq(WINDOW, 1 / sample_rate), BAND_EDGES_HZ) - 1
    levels = np.stack([spectrum[:, band == i].mean(axis=1) for i in range(LEDS_PER_EYE)], axis=1)
    peaks = np.percentile(levels, 98, axis=0)
    floor = peaks.max() * QUIET_BAND
    if floor <= 1e-6:
        return np.zeros_like(levels)
    levels = np.clip(levels / np.maximum(peaks, floor), 0, 1)
    for i in range(1, count):
        levels[i] = np.maximum(levels[i], levels[i - 1] * DECAY)
    return levels


def dimmed(rgb: int, level: float) -> int:
    return sum(round(((rgb >> shift) & 0xFF) * level) << shift for shift in (16, 8, 0))


def eye_frame(levels: np.ndarray, colour: int = COLOUR) -> list[int]:
    """Both eyes, left then right; the right ring is mirrored so the eyes look symmetric."""
    left = [dimmed(colour, float(level) ** GAMMA) for level in levels]
    return left + left[::-1]
