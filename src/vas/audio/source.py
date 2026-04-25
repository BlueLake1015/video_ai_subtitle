from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from ..types import AudioFrame


@runtime_checkable
class AudioSource(Protocol):
    """Async iterator of 16 kHz mono float32 audio frames with monotonic PTS."""

    async def frames(self) -> AsyncIterator[AudioFrame]: ...
    async def close(self) -> None: ...
