from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..types import Cue


@runtime_checkable
class Writer(Protocol):
    def render(self, cues: list[Cue]) -> str: ...

    def write(self, cues: list[Cue], path: str | Path) -> None: ...
