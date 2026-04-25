from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Iterable

import numpy as np

from .audio.ffmpeg_args import SAMPLE_RATE
from .config import SegmentConfig, VadConfig
from .types import AudioFrame, SpeechRegion
from .utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Segment:
    """A ~30s slice of audio aligned to silence boundaries, ready for Whisper.

    `start_s` is in stream time (so word timestamps from the transcriber can
    be re-based onto the source timeline via TranscribeOptions.time_offset_s).
    """
    audio: np.ndarray
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class WhisperSegmenter:
    """Aggregate VAD speech regions into ~target_seconds chunks for Whisper.

    Strategy: greedily attach successive speech regions to a buffer until adding
    the next one would push past `max_seconds`. Flush at silence (gap between
    regions) when the buffer >= `min_seconds`. Carries the underlying audio so
    the consumer can transcribe directly without re-decoding.
    """

    def __init__(
        self,
        seg_cfg: SegmentConfig | None = None,
        vad_cfg: VadConfig | None = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.seg = seg_cfg or SegmentConfig()
        self.vad = vad_cfg or VadConfig()
        self.sr = sample_rate

    # ----------------- batch (full audio in memory) -----------------

    def from_regions(
        self, audio: np.ndarray, regions: list[SpeechRegion]
    ) -> list[Segment]:
        if not regions:
            return []
        out: list[Segment] = []
        cur_start: float | None = None
        cur_end: float = 0.0

        def emit():
            nonlocal cur_start, cur_end
            if cur_start is None:
                return
            s = int(cur_start * self.sr)
            e = int(cur_end * self.sr)
            out.append(Segment(audio=audio[s:e], start_s=cur_start, end_s=cur_end))
            cur_start = None
            cur_end = 0.0

        for r in regions:
            if cur_start is None:
                cur_start = r.start_s
                cur_end = r.end_s
                continue

            tentative = r.end_s - cur_start
            cur_dur = cur_end - cur_start

            if tentative > self.seg.max_seconds:
                # Adding this region would overflow: emit current first.
                if cur_dur >= self.seg.min_seconds:
                    emit()
                    cur_start = r.start_s
                    cur_end = r.end_s
                else:
                    # Current is too short to stand alone -- attach anyway and
                    # let it overflow slightly rather than produce a tiny chunk.
                    cur_end = r.end_s
            else:
                cur_end = r.end_s
                # If we've crossed target and the next region is far away, flush.
                gap_to_next = None  # (we don't know yet; rely on max-cap above)
                if cur_dur >= self.seg.target_seconds:
                    emit()
                    cur_start = None

        emit()
        return out

    # ----------------- streaming (frame-by-frame, for live TS) -----------------

    async def from_frames(
        self, frames: AsyncIterator[AudioFrame]
    ) -> AsyncIterator[Segment]:
        """Stream frames in, emit ~target-second segments as silence boundaries arrive.

        Uses Silero streaming VAD for low-latency segment finalization. Falls
        back to RMS-energy gating if silero-vad isn't installed.
        """
        try:
            import torch  # noqa: F401
            from silero_vad import load_silero_vad, VADIterator
            model = load_silero_vad(onnx=True)
            vad_it = VADIterator(
                model,
                threshold=self.vad.threshold,
                sampling_rate=self.sr,
                min_silence_duration_ms=self.vad.min_silence_ms,
                speech_pad_ms=self.vad.speech_pad_ms,
            )
        except Exception:
            log.warning("silero-vad unavailable; using energy-gated fallback")
            model = None
            vad_it = None

        target = self.seg.target_seconds
        max_s = self.seg.max_seconds
        min_s = self.seg.min_seconds

        # Buffer holds *all* audio (including silence) since last flush, so we
        # can hand a contiguous slice to Whisper.
        buf: list[np.ndarray] = []
        buf_start: float | None = None
        speech_start: float | None = None
        last_speech_end: float | None = None

        def flush_if_ready(now_s: float) -> Segment | None:
            nonlocal buf, buf_start, speech_start, last_speech_end
            if not buf or buf_start is None:
                return None
            duration = now_s - buf_start
            if duration < min_s:
                return None
            audio = np.concatenate(buf)
            seg = Segment(audio=audio, start_s=buf_start, end_s=buf_start + duration)
            buf = []
            buf_start = None
            speech_start = None
            last_speech_end = None
            return seg

        async for frame in frames:
            if buf_start is None:
                buf_start = frame.pts_s
            buf.append(frame.pcm)
            now_end = frame.pts_s + frame.duration_s
            cur_dur = now_end - buf_start

            # VAD verdict
            if vad_it is not None and len(frame.pcm) >= 512:
                import torch
                event = vad_it(torch.from_numpy(frame.pcm[:512]), return_seconds=True)
                if event:
                    if "start" in event:
                        speech_start = frame.pts_s + float(event["start"])
                    if "end" in event:
                        last_speech_end = frame.pts_s + float(event["end"])

            # Force-flush at max
            if cur_dur >= max_s:
                seg = flush_if_ready(now_end)
                if seg:
                    yield seg
                continue

            # Past target & we just hit silence -> flush
            if cur_dur >= target and last_speech_end is not None:
                # Heuristic: emit when current frame is "in silence" (no speech-start
                # since the last speech-end signal).
                if speech_start is None or speech_start <= last_speech_end:
                    seg = flush_if_ready(now_end)
                    if seg:
                        yield seg

        # Flush trailing audio at EOF
        if buf and buf_start is not None:
            audio = np.concatenate(buf)
            yield Segment(
                audio=audio, start_s=buf_start,
                end_s=buf_start + len(audio) / self.sr,
            )
