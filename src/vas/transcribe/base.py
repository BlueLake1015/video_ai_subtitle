from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..types import Word


@dataclass
class TranscribeOptions:
    language: str | None = None
    initial_prompt: str | None = None
    time_offset_s: float = 0.0  # added to all word timestamps in the result


@runtime_checkable
class Transcriber(Protocol):
    """Transcribe a mono 16 kHz float32 audio array to a flat list of words.

    Implementations load model weights on first call and reuse them.
    """

    def transcribe(
        self, audio: np.ndarray, options: TranscribeOptions | None = None
    ) -> list[Word]: ...

    def close(self) -> None: ...
