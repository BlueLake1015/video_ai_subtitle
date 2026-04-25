from vas.audio.ffmpeg_args import build_args, _is_url, _is_mpegts_udp


def test_file_batch_mode():
    argv = build_args("/tmp/in.mp4", mode="batch")
    assert "-re" not in argv
    assert argv[-1] == "-"
    assert "-ar" in argv and "16000" in argv
    assert "-ac" in argv and "1" in argv
    assert "-f" in argv and "s16le" in argv


def test_file_realtime_adds_re():
    argv = build_args("/tmp/in.mp4", mode="realtime")
    assert "-re" in argv


def test_udp_mpegts_low_latency_flags():
    argv = build_args("udp://239.0.0.1:5000", mode="auto")
    assert "-fflags" in argv
    assert "nobuffer" in argv
    assert "-probesize" in argv
    assert "-fifo_size" in argv
    assert "-overrun_nonfatal" in argv


def test_rtp_mpegts_treated_same_as_udp():
    argv = build_args("rtp://host:5000", mode="auto")
    assert "-fifo_size" in argv


def test_url_detection():
    assert _is_url("udp://1.2.3.4:5000")
    assert _is_url("http://host/file.ts")
    assert not _is_url("/tmp/foo.ts")
    assert not _is_url("relative/path.mp4")


def test_mpegts_udp_detection():
    assert _is_mpegts_udp("udp://239.0.0.1:5000")
    assert _is_mpegts_udp("rtp://host:5000")
    assert not _is_mpegts_udp("http://host/file.ts")
