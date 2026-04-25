from vas.utils.timing import fmt_srt_time, fmt_vtt_time, fmt_ttml_time


def test_srt_time_format():
    assert fmt_srt_time(0) == "00:00:00,000"
    assert fmt_srt_time(1.5) == "00:00:01,500"
    assert fmt_srt_time(3661.234) == "01:01:01,234"


def test_vtt_time_format():
    assert fmt_vtt_time(0) == "00:00:00.000"
    assert fmt_vtt_time(1.5) == "00:00:01.500"


def test_ttml_matches_vtt():
    assert fmt_ttml_time(1.234) == fmt_vtt_time(1.234)


def test_negative_clamped_to_zero():
    assert fmt_srt_time(-1.0) == "00:00:00,000"
