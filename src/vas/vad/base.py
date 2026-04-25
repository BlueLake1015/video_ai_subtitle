from __future__ import annotations

from typing import AsyncIterator, Iterable, Protocol, runtime_checkable

import numpy as np

from ..types import AudioFrame, SpeechRegion


@runtime_checkable
class VAD(Protocol):
    """Batch VAD over a pre-decoded audio array."""

    def segment(self, audio: np.ndarray, sample_rate: int) -> list[SpeechRegion]: ...


@runtime_checkable
class StreamingVAD(Protocol):
    """Frame-by-frame VAD that emits speech regions as they finalize.

    Implementations are stateful. Yields a SpeechRegion when it is confident
    the region has ended (silence >= min_silence_ms).
    """

    async def regions(
        self, frames: AsyncIterator[AudioFrame]
    ) -> AsyncIterator[SpeechRegion]: ...
