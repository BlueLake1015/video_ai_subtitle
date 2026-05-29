from __future__ import annotations

import numpy as np

from ..config import TranscribeConfig
from ..types import Word
from ..utils.logging import get_logger
from .base import TranscribeOptions

log = get_logger(__name__)

SAMPLE_RATE = 16000


class WhisperXTranscriber:
    """WhisperX backend (faster-whisper ASR + wav2vec2 forced alignment).

    Requires the 'whisperx' extra:  pip install -e '.[whisperx]'

    This is a **self-contained** engine: it runs WhisperX's own VAD chunking,
    batched faster-whisper transcription, and phoneme forced-alignment over the
    whole file. The `consumes_full_audio` flag tells the batch pipeline to hand it
    the entire decoded audio and **skip this project's VAD + segmenter** (so we use
    WhisperX's pipeline rather than duplicating ours). Other backends are
    unaffected — they don't set this flag.

    The ASR runs on CTranslate2 (works regardless of the torch build); the
    alignment model is wav2vec2 on PyTorch, so it needs a torch build that supports
    the GPU's compute capability (or CPU). Default model is the CT2 build
    `Systran/faster-whisper-large-v3`. Alignment covers en/zh/ja/ko + ~40 langs.
    """

    consumes_full_audio = True

    def __init__(self, cfg: TranscribeConfig):
        self.cfg = cfg
        self._model = None
        self._whisperx = None
        self._device = "cpu"
        self._align_cache: dict[str, tuple] = {}
        self._batch_size = int(getattr(cfg, "batch_size", 16) or 16)

    def _ensure_model(self):
        if self._model is None:
            try:
                import whisperx
            except ImportError as e:
                raise ImportError(
                    "whisperx not installed. `pip install -e '.[whisperx]'`"
                ) from e
            self._whisperx = whisperx
            self._device = "cuda" if self.cfg.device in ("cuda", "auto") else "cpu"
            ref = self.cfg.model_path or self.cfg.model
            ct = self.cfg.compute_type or "float16"
            if self._device == "cpu" and ct not in ("int8", "float32"):
                ct = "int8"  # CTranslate2 has no fp16 CPU path
            log.info(
                "loading WhisperX model=%s device=%s compute_type=%s",
                ref, self._device, ct,
            )
            # Prefer Silero VAD (avoids the gated pyannote model); fall back if the
            # installed whisperx doesn't accept the kwarg.
            try:
                self._model = whisperx.load_model(
                    ref, self._device, compute_type=ct,
                    language=self.cfg.language, vad_method="silero",
                )
            except TypeError:
                self._model = whisperx.load_model(
                    ref, self._device, compute_type=ct, language=self.cfg.language,
                )
        return self._model

    def _aligner(self, lang: str):
        if lang not in self._align_cache:
            try:
                am, meta = self._whisperx.load_align_model(
                    language_code=lang, device=self._device
                )
                self._align_cache[lang] = (am, meta)
            except Exception as e:  # no alignment model for this language, etc.
                log.warning("WhisperX align model unavailable for %r: %s", lang, e)
                self._align_cache[lang] = (None, None)
        return self._align_cache[lang]

    def transcribe(
        self, audio: np.ndarray, options: TranscribeOptions | None = None
    ) -> list[Word]:
        model = self._ensure_model()
        opts = options or TranscribeOptions()
        off = opts.time_offset_s
        samples = np.asarray(audio, dtype=np.float32)

        result = model.transcribe(samples, batch_size=self._batch_size)
        segments = result.get("segments", [])
        lang = result.get("language") or opts.language or self.cfg.language

        # Forced alignment -> accurate word timestamps.
        words: list[Word] = []
        if lang and segments:
            align_model, metadata = self._aligner(lang)
            if align_model is not None:
                try:
                    aligned = self._whisperx.align(
                        segments, align_model, metadata, samples, self._device,
                        return_char_alignments=False,
                    )
                    for seg in aligned.get("segments", []):
                        for w in seg.get("words", []):
                            start, end = w.get("start"), w.get("end")
                            txt = str(w.get("word", "")).strip()
                            if start is None or end is None or not txt:
                                continue
                            words.append(Word(
                                text=txt,
                                start_s=float(start) + off,
                                end_s=float(end) + off,
                                probability=float(w.get("score", 1.0) or 1.0),
                            ))
                except Exception as e:
                    log.warning("WhisperX alignment failed: %s; using segment timings", e)

        if words:
            return words

        # Fallback: segment-level timings (no alignment available).
        out: list[Word] = []
        for seg in segments:
            txt = str(seg.get("text", "")).strip()
            if not txt or seg.get("start") is None or seg.get("end") is None:
                continue
            out.append(Word(
                text=txt,
                start_s=float(seg["start"]) + off,
                end_s=float(seg["end"]) + off,
                probability=1.0,
            ))
        return out

    def close(self) -> None:
        self._model = None
        self._align_cache = {}
