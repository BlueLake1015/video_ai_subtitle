from __future__ import annotations

import numpy as np

from ..config import TranscribeConfig
from ..types import Word
from ..utils.logging import get_logger
from .base import TranscribeOptions

log = get_logger(__name__)

SAMPLE_RATE = 16000


class OpenAiWhisperTranscriber:
    """Original OpenAI Whisper backend — the reference PyTorch implementation.

    Requires the 'openai-whisper' extra:  pip install -e '.[openai-whisper]'

    Distinct from the default `faster_whisper` backend (CTranslate2) and
    `whisper_cpp`: it uses the `whisper` package's own decoding and word-timestamp
    alignment. Language is an ISO code (same as the rest of the project) or null to
    auto-detect. Runs on PyTorch, so it needs a torch build that supports the GPU's
    compute capability (or run on CPU via `--asr-device cpu`).
    """

    def __init__(self, cfg: TranscribeConfig):
        self.cfg = cfg
        self._model = None
        self._device = "cpu"

    def _resolve_device(self) -> str:
        if self.cfg.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.cfg.device

    def _ensure_model(self):
        if self._model is None:
            try:
                import whisper
            except ImportError as e:
                raise ImportError(
                    "openai-whisper not installed. `pip install -e '.[openai-whisper]'`"
                ) from e
            self._device = self._resolve_device()
            ref = self.cfg.model_path or self.cfg.model
            log.info("loading openai-whisper model=%s device=%s", ref, self._device)
            self._model = whisper.load_model(ref, device=self._device)
        return self._model

    def transcribe(
        self, audio: np.ndarray, options: TranscribeOptions | None = None
    ) -> list[Word]:
        model = self._ensure_model()
        opts = options or TranscribeOptions()
        lang = opts.language or self.cfg.language
        prompt = opts.initial_prompt or self.cfg.initial_prompt
        # fp16 only on CUDA; whisper warns + falls back to fp32 on CPU otherwise.
        fp16 = self._device == "cuda" and "float16" in (self.cfg.compute_type or "")

        kwargs = {
            "language": lang,
            "task": "transcribe",
            "temperature": self.cfg.temperature,
            "condition_on_previous_text": self.cfg.condition_on_previous_text,
            "word_timestamps": self.cfg.word_timestamps,
            "initial_prompt": prompt,
            "fp16": fp16,
            "verbose": False,
        }
        if self.cfg.beam_size and self.cfg.beam_size > 1:
            kwargs["beam_size"] = self.cfg.beam_size
        # best_of only applies to sampling (temperature > 0); ignored under beam search.
        if self.cfg.best_of and self.cfg.best_of > 1 and self.cfg.temperature > 0:
            kwargs["best_of"] = self.cfg.best_of

        result = model.transcribe(np.asarray(audio, dtype=np.float32), **kwargs)

        words: list[Word] = []
        off = opts.time_offset_s
        for seg in result.get("segments", []):
            seg_words = seg.get("words") if self.cfg.word_timestamps else None
            if seg_words:
                for w in seg_words:
                    words.append(Word(
                        text=str(w.get("word", "")).strip(),
                        start_s=float(w.get("start", seg["start"])) + off,
                        end_s=float(w.get("end", seg["end"])) + off,
                        probability=float(w.get("probability", 1.0) or 1.0),
                    ))
            else:
                words.append(Word(
                    text=str(seg.get("text", "")).strip(),
                    start_s=float(seg["start"]) + off,
                    end_s=float(seg["end"]) + off,
                    probability=1.0,
                ))
        return [w for w in words if w.text]

    def close(self) -> None:
        self._model = None
