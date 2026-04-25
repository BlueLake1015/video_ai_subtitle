from vas.subtitle_io import read_cues
from vas.writers import SrtWriter
from vas.types import Cue


def test_roundtrip_srt(tmp_path):
    cues_in = [
        Cue(index=1, start_s=0.0, end_s=1.5, lines=["hello world"]),
        Cue(index=2, start_s=2.0, end_s=3.5, lines=["goodbye", "world"]),
    ]
    p = tmp_path / "in.srt"
    SrtWriter().write(cues_in, p)
    cues_out = read_cues(p)
    assert len(cues_out) == 2
    assert cues_out[0].text == "hello world"
    assert cues_out[1].lines == ["goodbye", "world"]
    assert abs(cues_out[0].start_s - 0.0) < 1e-3
    assert abs(cues_out[1].end_s - 3.5) < 1e-3


def test_reads_dot_or_comma_ms(tmp_path):
    p = tmp_path / "x.srt"
    p.write_text(
        "1\n00:00:01.000 --> 00:00:02.500\nfoo\n\n"
        "2\n00:00:03,000 --> 00:00:04,500\nbar\n",
        encoding="utf-8",
    )
    cues = read_cues(p)
    assert [c.text for c in cues] == ["foo", "bar"]
