from __future__ import annotations

from pathlib import Path

from ..types import Cue
from ..utils.timing import fmt_vtt_time


class VttWriter:
    def render(self, cues: list[Cue]) -> str:
        out: list[str] = ["WEBVTT", ""]
        for c in cues:
            out.append(f"{fmt_vtt_time(c.start_s)} --> {fmt_vtt_time(c.end_s)}")
            out.append(c.text)
            out.append("")
        return "\n".join(out) + ("\n" if out else "")

    def write(self, cues: list[Cue], path: str | Path) -> None:
        Path(path).write_text(self.render(cues), encoding="utf-8")
