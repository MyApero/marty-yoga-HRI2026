import cv2

from src.window import SUBTITLE_FONT, SUBTITLE_MAX_LINES, fit_subtitles, wrap_text

SENTENCE = "Lift your left arm a little higher and keep your shoulders relaxed while you breathe slowly."


def width(line, scale):
    return cv2.getTextSize(line, SUBTITLE_FONT, scale, 2)[0][0]


def test_wrapping_keeps_every_word_in_order_within_the_width():
    lines = wrap_text(SENTENCE, 1.8, 2, 800)
    assert len(lines) > 1
    assert " ".join(lines) == SENTENCE
    assert all(width(line, 1.8) <= 800 for line in lines)


def test_a_long_text_shrinks_the_font_to_fit_the_band():
    text = " ".join([SENTENCE] * 3)
    lines, scale = fit_subtitles(text, 1.8, 2, 2000)
    assert scale < 1.8
    assert len(lines) <= SUBTITLE_MAX_LINES
    assert " ".join(lines) == text


def test_a_short_sentence_stays_on_one_line_at_full_size():
    assert fit_subtitles("Great job.", 1.8, 2, 2000) == (["Great job."], 1.8)
