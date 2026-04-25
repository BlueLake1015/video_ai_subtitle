from __future__ import annotations

import re
from pathlib import Path

from .types import Cue


_SRT_BLOCK_TIME = re.compile(
    r"(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{3})"
)


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def read_cues(path: str | Path) -> list[Cue]:
    """Parse an SRT or VTT file into Cues. Tolerates both `,` and `.` for ms."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    cues: list[Cue] = []
    idx = 1
    for block in blocks:
        m = _SRT_BLOCK_TIME.search(block)
        if not m:
            continue
        start = _to_seconds(m["sh"], m["sm"], m["ss"], m["sms"])
        end = _to_seconds(m["eh"], m["em"], m["es"], m["ems"])
        # Lines after the time line are the text body.
        lines: list[str] = []
        seen_time = False
        for line in block.splitlines():
            if not seen_time:
                if _SRT_BLOCK_TIME.search(line):
                    seen_time = True
                continue
            if line.strip():
                lines.append(line.strip())
        if not lines:
            continue
        cues.append(Cue(index=idx, start_s=start, end_s=end, lines=lines))
        idx += 1
    return cues
