from __future__ import annotations

from urllib.parse import urlparse

SAMPLE_RATE = 16000
CHANNELS = 1
PCM_FMT = "s16le"


def _scheme(input_: str) -> str:
    return urlparse(input_).scheme.lower()


def _is_url(input_: str) -> bool:
    """Anything with a scheme we treat as a network input."""
    return _scheme(input_) in {"udp", "rtp", "tcp", "http", "https", "srt"}


def _is_mpegts_udp(input_: str) -> bool:
    """MPEG2-TS over UDP / RTP -- the project's primary live ingest target."""
    return _scheme(input_) in {"udp", "rtp"}


def build_args(
    input_: str,
    *,
    mode: str = "auto",  # "auto" | "batch" | "realtime"
    extra_input_flags: list[str] | None = None,
) -> list[str]:
    """Build the ffmpeg argv (without the leading 'ffmpeg') for audio extraction.

    Project scope: local file *or* MPEG2-TS stream (typically `udp://239.x.y.z:port`).
    For local files, default to batch (decode as fast as possible). For URLs,
    default to realtime (let ffmpeg pace the read to wall-clock).
    """
    is_url = _is_url(input_)
    is_ts_udp = _is_mpegts_udp(input_)

    if mode == "auto":
        mode = "realtime" if is_url else "batch"

    pre = ["-hide_banner", "-loglevel", "error", "-nostdin"]

    if is_url:
        # Low-latency demuxer flags for live transport streams.
        # MPEG2-TS needs a non-zero probesize to discover PAT/PMT stream tables;
        # 32 bytes (which works for some elementary streams) leaves the demuxer
        # reporting "Output file #0 does not contain any stream". Use values
        # large enough to read the first ~100 TS packets but still low-latency.
        if is_ts_udp:
            probesize = "500000"        # ~500 KB, well above one PAT/PMT cycle
            analyzeduration = "200000"  # 200 ms
        else:
            probesize = "32"
            analyzeduration = "0"
        pre += [
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", probesize,
            "-analyzeduration", analyzeduration,
        ]

    if is_ts_udp:
        # UDP buffer tuning: default kernel buffer drops packets on bursty TS.
        # `fifo_size` is in 188-byte TS packet units; 1MB ~= 5570.
        # `overrun_nonfatal=1` keeps reading instead of aborting on overflow.
        pre += ["-fifo_size", "1000000", "-overrun_nonfatal", "1"]

    if mode == "realtime" and not is_url:
        # Pace a local file to wall-clock to simulate a live source.
        pre += ["-re"]

    if extra_input_flags:
        pre += list(extra_input_flags)

    return [
        *pre,
        "-i", input_,
        "-vn",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-f", PCM_FMT,
        "-acodec", "pcm_s16le",
        "-",
    ]
