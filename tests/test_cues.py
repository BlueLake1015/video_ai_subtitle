from vas.config import CueConfig
from vas.cues.assembler import CueAssembler
from vas.types import Word


def _words(s: str, start: float = 0.0, step: float = 0.3) -> list[Word]:
    out = []
    t = start
    for tok in s.split():
        out.append(Word(text=tok, start_s=t, end_s=t + step, probability=1.0))
        t += step
    return out


def test_simple_single_cue():
    ws = _words("hello world")
    cues = CueAssembler().assemble(ws)
    assert len(cues) == 1
    assert cues[0].text == "hello world"
    assert cues[0].start_s == 0.0
    assert cues[0].end_s == 0.6


def test_sentence_terminator_splits_cue():
    ws = _words("hello world. goodbye world")
    # Manually fix: make "world." the 2nd token
    ws = [
        Word("hello", 0.0, 0.3),
        Word("world.", 0.3, 0.6),
        Word("goodbye", 0.7, 1.0),
        Word("world", 1.0, 1.3),
    ]
    cues = CueAssembler().assemble(ws)
    assert len(cues) == 2
    assert cues[0].text.startswith("hello")
    assert cues[1].text.startswith("goodbye")


def test_max_duration_splits_long_run():
    cfg = CueConfig(max_duration_s=2.0, max_chars_per_line=80, max_lines=2)
    ws = [Word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(10)]
    cues = CueAssembler(cfg).assemble(ws)
    assert len(cues) >= 2
    for c in cues:
        assert c.duration_s <= 2.5  # some slack for gap enforcement


def test_line_wrap_respects_max_chars():
    cfg = CueConfig(max_chars_per_line=10, max_lines=3, max_duration_s=60.0)
    ws = _words("one two three four five six seven")
    cues = CueAssembler(cfg).assemble(ws)
    assert cues
    for c in cues:
        for line in c.lines:
            assert len(line) <= cfg.max_chars_per_line


def test_min_gap_enforced():
    # Words spaced 100 ms apart; min_gap=200ms forces shrinking the previous cue.
    cfg = CueConfig(min_gap_ms=200, max_duration_s=0.6, max_chars_per_line=80, max_lines=2)
    ws = [
        Word("a", 0.0, 0.5),
        Word("b", 0.6, 1.0),
        Word("c", 1.1, 1.5),
    ]
    cues = CueAssembler(cfg).assemble(ws)
    assert len(cues) >= 2
    for a, b in zip(cues, cues[1:]):
        assert b.start_s - a.end_s >= 0.2 - 1e-6


def test_empty_input():
    assert CueAssembler().assemble([]) == []
