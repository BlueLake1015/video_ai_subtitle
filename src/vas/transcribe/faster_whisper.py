from __future__ import annotations

import numpy as np

from ..config import TranscribeConfig
from ..types import Word
from ..utils.logging import get_logger
from .base import TranscribeOptions

log = get_logger(__name__)


class FasterWhisperTranscriber:
    def __init__(self, cfg: TranscribeConfig):
        self.cfg = cfg
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            log.info(
                "loading faster-whisper model=%s device=%s compute_type=%s",
                self.cfg.model, self.cfg.device, self.cfg.compute_type,
            )
            self._model = WhisperModel(
                self.cfg.model_path or self.cfg.model,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
        return self._model

    def transcribe(
        self, audio: np.ndarray, options: TranscribeOptions | None = None
    ) -> list[Word]:
        model = self._ensure_model()
        opts = options or TranscribeOptions()
        lang = opts.language or self.cfg.language
        prompt = opts.initial_prompt or self.cfg.initial_prompt

        segments, _info = model.transcribe(
            audio.astype(np.float32),
            language=lang,
            beam_size=self.cfg.beam_size,
            best_of=self.cfg.best_of,
            temperature=self.cfg.temperature,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            word_timestamps=self.cfg.word_timestamps,
            initial_prompt=prompt,
            vad_filter=False,  # we run VAD externally
        )

        words: list[Word] = []
        off = opts.time_offset_s
        for seg in segments:
            if self.cfg.word_timestamps and seg.words:
                for w in seg.words:
                    words.append(Word(
                        text=w.word,
                        start_s=(w.start or seg.start) + off,
                        end_s=(w.end or seg.end) + off,
                        probability=getattr(w, "probability", 1.0) or 1.0,
                    ))
            else:
                words.append(Word(
                    text=seg.text,
                    start_s=seg.start + off,
                    end_s=seg.end + off,
                    probability=1.0,
                ))
        return words

    def close(self) -> None:
        self._model = None
