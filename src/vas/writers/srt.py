from __future__ import annotations

from pathlib import Path

from ..types import Cue
from ..utils.timing import fmt_srt_time


class SrtWriter:
    def render(self, cues: list[Cue]) -> str:
        out: list[str] = []
        for i, c in enumerate(cues, start=1):
            out.append(str(i))
            out.append(f"{fmt_srt_time(c.start_s)} --> {fmt_srt_time(c.end_s)}")
            out.append(c.text)
            out.append("")
        return "\n".join(out) + ("\n" if out else "")

    def write(self, cues: list[Cue], path: str | Path) -> None:
        Path(path).write_text(self.render(cues), encoding="utf-8")
