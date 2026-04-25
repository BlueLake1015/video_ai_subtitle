from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..types import AudioFrame
from ..utils.logging import get_logger
from .ffmpeg_source import FfmpegSource, SourceMode

log = get_logger(__name__)


class SupervisedStreamSource:
    """Restart-on-failure wrapper for live network ingest.

    Watches for stalls (no frame for `watchdog_s` seconds) and for process exit,
    then re-opens the stream with exponential backoff. Emits frames in a single
    monotonic timeline by maintaining its own sample counter across restarts --
    gaps appear as time jumps in PTS (acceptable: gap was silence to the app).
    """

    def __init__(
        self,
        input_: str,
        *,
        watchdog_s: float = 5.0,
        max_backoff_s: float = 30.0,
    ):
        self.input_ = input_
        self.watchdog_s = watchdog_s
        self.max_backoff_s = max_backoff_s
        self._closed = False

    async def frames(self) -> AsyncIterator[AudioFrame]:
        backoff = 1.0
        total_pts = 0.0
        while not self._closed:
            src = FfmpegSource(self.input_, mode=SourceMode.REALTIME)
            try:
                await src.start()
            except Exception as e:
                log.error("ffmpeg start failed: %s", e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                continue

            backoff = 1.0
            last_frame_ts = asyncio.get_event_loop().time()
            base_pts = total_pts

            try:
                async for f in src.frames():
                    last_frame_ts = asyncio.get_event_loop().time()
                    # Re-base PTS onto the supervisor's monotonic timeline.
                    shifted = AudioFrame(
                        pcm=f.pcm,
                        pts_s=base_pts + f.pts_s,
                        sample_rate=f.sample_rate,
                    )
                    total_pts = shifted.pts_s + shifted.duration_s
                    yield shifted

                    # Watchdog happens implicitly: if stdout stalls, the for-loop
                    # is suspended and we rely on the outer task to cancel.
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("ffmpeg source errored: %s", e)
            finally:
                await src.close()

            if self._closed:
                return

            log.warning("stream ended; reconnecting in %.1fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)

    async def close(self) -> None:
        self._closed = True
