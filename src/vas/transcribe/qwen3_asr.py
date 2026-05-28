from __future__ import annotations

import tempfile
import wave

import numpy as np

from ..config import TranscribeConfig
from ..types import Word
from ..utils.logging import get_logger
from .base import TranscribeOptions

log = get_logger(__name__)

# Qwen3-ASR expects a language NAME (or None to auto-detect), not an ISO code.
# Map the common ISO 639-1 codes; unknown codes fall back to auto-detect.
_ISO_TO_QWEN_LANG = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish",
    "cs": "Czech", "fa": "Persian", "el": "Greek", "hu": "Hungarian", "ro": "Romanian",
}

# compute_type -> torch dtype name. Qwen3-ASR has no int8 path; map to bf16.
_DTYPE_NAME = {
    "float16": "float16", "bfloat16": "bfloat16", "float32": "float32",
    "int8_float16": "bfloat16", "int8": "bfloat16",
}

SAMPLE_RATE = 16000


class Qwen3AsrTranscriber:
    """Qwen3-ASR backend (Alibaba), via the `qwen-asr` package.

    Requires the 'qwen' extra:  pip install -e '.[qwen]'

    Qwen3-ASR returns plain text. Real word-level timestamps need the optional
    forced-aligner model — set `forced_aligner: Qwen/Qwen3-ForcedAligner-0.6B` in
    the preset. Without it, word timings are interpolated across each ~30 s VAD
    segment (proportional to token length); the cue assembler still breaks those
    into readable cues, but the per-word timing is approximate.
    """

    def __init__(self, cfg: TranscribeConfig):
        self.cfg = cfg
        self._model = None
        self._aligner = getattr(cfg, "forced_aligner", None) or None
        self._use_ts = bool(self._aligner)

    def _ensure_model(self):
        if self._model is None:
            try:
                import torch
                from qwen_asr import Qwen3ASRModel
            except ImportError as e:
                raise ImportError(
                    "qwen-asr not installed. `pip install -e '.[qwen]'`"
                ) from e
            dtype = getattr(torch, _DTYPE_NAME.get(self.cfg.compute_type, "bfloat16"))
            device_map = "cuda:0" if self.cfg.device in ("cuda", "auto") else "cpu"
            kwargs = {
                "dtype": dtype,
                "device_map": device_map,
                "max_new_tokens": getattr(self.cfg, "max_new_tokens", 256),
            }
            if self._aligner:
                kwargs["forced_aligner"] = self._aligner
                kwargs["forced_aligner_kwargs"] = {"dtype": dtype, "device_map": device_map}
            log.info(
                "loading Qwen3-ASR model=%s device=%s dtype=%s aligner=%s",
                self.cfg.model, device_map, dtype, self._aligner or "none",
            )
            self._model = Qwen3ASRModel.from_pretrained(self.cfg.model, **kwargs)
        return self._model

    def _qwen_language(self, opts: TranscribeOptions) -> str | None:
        code = opts.language or self.cfg.language
        if not code:
            return None
        return _ISO_TO_QWEN_LANG.get(code.lower())

    def transcribe(
        self, audio: np.ndarray, options: TranscribeOptions | None = None
    ) -> list[Word]:
        model = self._ensure_model()
        opts = options or TranscribeOptions()
        off = opts.time_offset_s
        samples = np.asarray(audio, dtype=np.float32)
        duration = len(samples) / SAMPLE_RATE

        # qwen-asr's transcribe() most reliably accepts a file path; write the
        # segment to a temp 16 kHz mono WAV.
        with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
            _write_wav16(tf.name, samples)
            result = model.transcribe(
                audio=tf.name,
                language=self._qwen_language(opts),
                return_time_stamps=self._use_ts,
            )

        r = result[0] if isinstance(result, (list, tuple)) else result
        text = (getattr(r, "text", None) or "").strip()
        if not text:
            return []

        if self._use_ts:
            words = _words_from_timestamps(getattr(r, "time_stamps", None), off)
            if words:
                # The aligner emits bare words (no punctuation); the full text has
                # it. The two are 1:1, so reattach punctuation onto the timed words.
                return _reattach_punctuation(words, text)
            log.warning("Qwen3-ASR returned no parseable timestamps; interpolating")
        return _interpolate_words(text, off, duration)

    def close(self) -> None:
        self._model = None


def _write_wav16(path: str, samples: np.ndarray) -> None:
    """Write a float32 [-1, 1] mono array as a 16 kHz signed-16-bit PCM WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _interpolate_words(text: str, offset: float, duration: float) -> list[Word]:
    """Spread word timings across [offset, offset+duration], weighted by token
    length. Approximate, but enough for the cue assembler to break readable cues."""
    tokens = text.split()
    if not tokens:
        return []
    weights = [len(t) + 1 for t in tokens]
    total = float(sum(weights))
    words: list[Word] = []
    acc = 0.0
    for tok, wgt in zip(tokens, weights):
        start_frac = acc / total
        acc += wgt
        end_frac = acc / total
        words.append(Word(
            text=tok,
            start_s=offset + start_frac * duration,
            end_s=offset + end_frac * duration,
            probability=1.0,
        ))
    return words


def _reattach_punctuation(words: list[Word], text: str) -> list[Word]:
    """Carry punctuation/casing from the full transcript onto the aligner's timed
    words. The aligner strips punctuation but its word sequence matches the text
    1:1; when the token counts agree, swap in the punctuated tokens (keeping the
    timings). On any mismatch, return the timed words unchanged."""
    tokens = text.split()
    if len(tokens) != len(words):
        return words
    return [
        Word(text=tok, start_s=w.start_s, end_s=w.end_s, probability=w.probability)
        for w, tok in zip(words, tokens)
    ]


def _words_from_timestamps(time_stamps, offset: float) -> list[Word]:
    """Map the forced aligner's timestamps to Words. Defensive about shape:
    accepts ForcedAlignItem-style objects (.text/.start_time/.end_time), dicts
    ({word/text, start/end}), or (text, start, end) tuples; returns [] on any
    unexpected structure so the caller can fall back to interpolation."""
    if not time_stamps:
        return []
    out: list[Word] = []
    try:
        for item in time_stamps:
            if isinstance(item, dict):
                txt = item.get("word") or item.get("text") or item.get("token") or ""
                start = item.get("start", item.get("t0"))
                end = item.get("end", item.get("t1"))
            elif hasattr(item, "start_time") and hasattr(item, "end_time"):
                # qwen_asr ForcedAlignItem(text, start_time, end_time)
                txt = getattr(item, "text", "") or getattr(item, "word", "")
                start = item.start_time
                end = item.end_time
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                txt, start, end = item[0], item[1], item[2]
            else:
                return []
            if start is None or end is None or not str(txt).strip():
                continue
            out.append(Word(
                text=str(txt).strip(),
                start_s=float(start) + offset,
                end_s=float(end) + offset,
                probability=1.0,
            ))
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        return []
    return out
