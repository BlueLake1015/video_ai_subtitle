from __future__ import annotations


def fmt_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = round(seconds * 1000)
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = round(seconds * 1000)
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fmt_ttml_time(seconds: float) -> str:
    """TTML uses '00:00:00.000' clock-time format."""
    return fmt_vtt_time(seconds)
