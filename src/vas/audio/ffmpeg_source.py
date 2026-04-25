from __future__ import annotations

import asyncio
import shutil
from enum import Enum
from typing import AsyncIterator

import numpy as np

from ..types import AudioFrame
from ..utils.logging import get_logger
from .ffmpeg_args import SAMPLE_RATE, build_args

log = get_logger(__name__)

FRAME_SAMPLES = 512  # 32 ms @ 16 kHz, matches Silero VAD window
FRAME_BYTES = FRAME_SAMPLES * 2  # s16le = 2 bytes/sample
INT16_TO_FLOAT = 1.0 / 32768.0


class SourceMode(str, Enum):
    AUTO = "auto"
    BATCH = "batch"
    REALTIME = "realtime"


class FfmpegSource:
    """Unified audio source backed by an ffmpeg subprocess producing s16le PCM on stdout.

    Use for both local files (batch or realtime-paced) and network streams
    (RTMP / RTSP / SRT / HLS / HTTP). Emits AudioFrame objects every 32 ms.
    """

    def __init__(
        self,
        input_: str,
        *,
        mode: SourceMode | str = SourceMode.AUTO,
        ffmpeg_bin: str | None = None,
        extra_input_flags: list[str] | None = None,
    ):
        self.input_ = input_
        self.mode = SourceMode(mode) if isinstance(mode, str) else mode
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        self.extra_input_flags = extra_input_flags
        self._proc: asyncio.subprocess.Process | None = None
        self._samples_read = 0
        self._stderr_task: asyncio.Task | None = None

    async def start(self) -> None:
        argv = build_args(
            self.input_,
            mode=self.mode.value,
            extra_input_flags=self.extra_input_flags,
        )
        log.info("ffmpeg %s", " ".join(argv))
        self._proc = await asyncio.create_subprocess_exec(
            self.ffmpeg_bin, *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                return
            try:
                log.warning("ffmpeg: %s", line.decode(errors="replace").rstrip())
            except Exception:
                pass

    async def frames(self) -> AsyncIterator[AudioFrame]:
        if self._proc is None:
            await self.start()
        assert self._proc and self._proc.stdout

        while True:
            buf = await self._proc.stdout.readexactly(FRAME_BYTES) if False else None
            try:
                buf = await _read_exact(self._proc.stdout, FRAME_BYTES)
            except asyncio.IncompleteReadError as e:
                if e.partial:
                    pcm = _decode_s16le(e.partial)
                    pts = self._samples_read / SAMPLE_RATE
                    self._samples_read += len(pcm)
                    yield AudioFrame(pcm=pcm, pts_s=pts, sample_rate=SAMPLE_RATE)
                return

            pcm = _decode_s16le(buf)
            pts = self._samples_read / SAMPLE_RATE
            self._samples_read += len(pcm)
            yield AudioFrame(pcm=pcm, pts_s=pts, sample_rate=SAMPLE_RATE)

    async def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass


async def _read_exact(stream: asyncio.StreamReader, n: int) -> bytes:
    """Like StreamReader.readexactly but works around short reads on pipes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = await stream.read(n - len(buf))
        if not chunk:
            raise asyncio.IncompleteReadError(bytes(buf), n)
        buf.extend(chunk)
    return bytes(buf)


def _decode_s16le(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32) * INT16_TO_FLOAT
    return arr
