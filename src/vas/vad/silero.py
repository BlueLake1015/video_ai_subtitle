from __future__ import annotations

from typing import AsyncIterator

import numpy as np

from ..config import VadConfig
from ..types import AudioFrame, SpeechRegion
from ..utils.logging import get_logger

log = get_logger(__name__)

# Silero wants 16 kHz, 512-sample windows for int16 path; the pip package
# exposes `load_silero_vad` and `get_speech_timestamps` for batch and an
# `VADIterator` class for streaming.


class SileroBatchVAD:
    """Batch VAD using silero-vad's get_speech_timestamps."""

    def __init__(self, cfg: VadConfig | None = None):
        self.cfg = cfg or VadConfig()
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
        return self._model

    def segment(self, audio: np.ndarray, sample_rate: int) -> list[SpeechRegion]:
        from silero_vad import get_speech_timestamps
        import torch

        if sample_rate != 16000:
            raise ValueError(f"SileroBatchVAD expects 16 kHz audio, got {sample_rate}")
        model = self._ensure_model()
        t = torch.from_numpy(audio)
        ts = get_speech_timestamps(
            t,
            model,
            sampling_rate=sample_rate,
            threshold=self.cfg.threshold,
            min_speech_duration_ms=self.cfg.min_speech_ms,
            min_silence_duration_ms=self.cfg.min_silence_ms,
            speech_pad_ms=self.cfg.speech_pad_ms,
        )
        return [
            SpeechRegion(start_s=r["start"] / sample_rate, end_s=r["end"] / sample_rate)
            for r in ts
        ]


class SileroStreamingVAD:
    """Streaming VAD via silero-vad's VADIterator on 32 ms frames."""

    def __init__(self, cfg: VadConfig | None = None):
        self.cfg = cfg or VadConfig()
        self._iter = None

    def _ensure_iter(self):
        if self._iter is None:
            from silero_vad import load_silero_vad, VADIterator
            model = load_silero_vad(onnx=True)
            self._iter = VADIterator(
                model,
                threshold=self.cfg.threshold,
                sampling_rate=16000,
                min_silence_duration_ms=self.cfg.min_silence_ms,
                speech_pad_ms=self.cfg.speech_pad_ms,
            )
        return self._iter

    async def regions(
        self, frames: AsyncIterator[AudioFrame]
    ) -> AsyncIterator[SpeechRegion]:
        import torch
        vit = self._ensure_iter()
        current_start: float | None = None

        async for frame in frames:
            # Silero expects a torch tensor of exactly 512 samples at 16 kHz.
            if len(frame.pcm) < 512:
                continue
            t = torch.from_numpy(frame.pcm[:512])
            event = vit(t, return_seconds=True)
            if event:
                if "start" in event:
                    current_start = frame.pts_s + event["start"]
                if "end" in event and current_start is not None:
                    end_s = frame.pts_s + event["end"]
                    yield SpeechRegion(start_s=current_start, end_s=end_s)
                    current_start = None

        if current_start is not None:
            # Flush on EOF: treat end of stream as end of current region.
            yield SpeechRegion(start_s=current_start, end_s=current_start + 0.032)
