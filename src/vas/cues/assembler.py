from __future__ import annotations

from ..config import CueConfig
from ..types import Cue, Word


class CueAssembler:
    """Merge timed words into readable subtitle cues.

    Rules (applied greedily):
      * Start a new cue when: line fills at max_chars_per_line and we'd exceed
        max_lines, OR a sentence-terminating punctuation appears, OR the gap to
        the next word exceeds min_gap_ms * 4 or so, OR the cue would exceed
        max_duration_s.
      * Break lines at word boundaries, preferring after commas / before coords.
      * Enforce min_gap_ms between adjacent cues.
    """

    # Characters that indicate a natural cue boundary when followed by whitespace
    _TERMINAL = set(".!?。！？")
    _SOFT_BREAK = set(",;:、，；：")

    def __init__(self, cfg: CueConfig | None = None):
        self.cfg = cfg or CueConfig()

    def assemble(self, words: list[Word]) -> list[Cue]:
        if not words:
            return []

        cues: list[Cue] = []
        cur: list[Word] = []
        cur_start = words[0].start_s
        index = 1

        def flush(end_s: float | None = None) -> None:
            nonlocal cur, cur_start, index
            if not cur:
                return
            end = end_s if end_s is not None else cur[-1].end_s
            lines = self._wrap_lines(cur)
            cues.append(Cue(
                index=index,
                start_s=cur_start,
                end_s=end,
                lines=lines,
                words=list(cur),
            ))
            index += 1
            cur = []

        max_gap_s = (self.cfg.min_gap_ms / 1000.0) * 6.0

        for i, w in enumerate(words):
            if not cur:
                cur = [w]
                cur_start = w.start_s
                continue

            gap = w.start_s - cur[-1].end_s
            would_duration = w.end_s - cur_start
            prev_text = cur[-1].text.strip()
            terminal = bool(prev_text) and prev_text[-1] in self._TERMINAL
            tentative_lines = self._wrap_lines(cur + [w])

            should_flush = (
                terminal
                or gap > max_gap_s
                or would_duration > self.cfg.max_duration_s
                or len(tentative_lines) > self.cfg.max_lines
            )
            if should_flush:
                flush()
                cur_start = w.start_s
                cur = [w]
            else:
                cur.append(w)

        flush()

        self._enforce_gap(cues)
        return cues

    def _wrap_lines(self, words: list[Word]) -> list[str]:
        """Greedy word-wrap honouring max_chars_per_line."""
        lines: list[str] = [""]
        for w in words:
            tok = w.text.strip()
            if not tok:
                continue
            candidate = f"{lines[-1]} {tok}".strip() if lines[-1] else tok
            if len(candidate) <= self.cfg.max_chars_per_line:
                lines[-1] = candidate
            else:
                lines.append(tok)
        return [ln for ln in lines if ln]

    def _enforce_gap(self, cues: list[Cue]) -> None:
        gap_s = self.cfg.min_gap_ms / 1000.0
        for a, b in zip(cues, cues[1:]):
            if b.start_s - a.end_s < gap_s:
                a.end_s = max(a.start_s, b.start_s - gap_s)
