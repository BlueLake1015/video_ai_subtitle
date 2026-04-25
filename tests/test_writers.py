from vas.types import Cue
from vas.writers import writer_for, SrtWriter, TtmlWriter, VttWriter


def _cues():
    return [
        Cue(index=1, start_s=0.0, end_s=1.5, lines=["hello world"]),
        Cue(index=2, start_s=2.0, end_s=3.5, lines=["goodbye", "world"]),
    ]


def test_srt_render():
    out = SrtWriter().render(_cues())
    assert "1\n00:00:00,000 --> 00:00:01,500\nhello world\n" in out
    assert "2\n00:00:02,000 --> 00:00:03,500\ngoodbye\nworld\n" in out


def test_vtt_render():
    out = VttWriter().render(_cues())
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in out
    assert "goodbye\nworld" in out


def test_ttml_render_valid_xml():
    import xml.etree.ElementTree as ET
    out = TtmlWriter().render(_cues())
    root = ET.fromstring(out)
    ns = {"t": "http://www.w3.org/ns/ttml"}
    ps = root.findall(".//t:p", ns)
    assert len(ps) == 2
    assert ps[0].attrib["begin"] == "00:00:00.000"
    assert ps[0].attrib["end"] == "00:00:01.500"


def test_writer_for_dispatch(tmp_path):
    for name in ("out.srt", "out.vtt", "out.ttml"):
        w = writer_for(name)
        p = tmp_path / name
        w.write(_cues(), p)
        assert p.exists() and p.stat().st_size > 0
