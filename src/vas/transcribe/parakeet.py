from __future__ import annotations

import tempfile
import wave

import numpy as np

from ..config import TranscribeConfig
from ..types import Word
from ..utils.logging import get_logger
from .base import TranscribeOptions

log = get_logger(__name__)

SAMPLE_RATE = 16000


class ParakeetTranscriber:
    """NVIDIA Parakeet-TDT backend via NeMo (`nemo_toolkit[asr]`).

    Requires the 'parakeet' extra:  pip install -e '.[parakeet]'

    FastConformer-TDT — tops the Open ASR leaderboard on English speed/accuracy.
    Supports English + 24 European languages (auto-detected); does **not** support
    Chinese / Japanese / Korean. Output is punctuated and capitalized, with
    word-level timestamps. Runs on PyTorch, so it needs a torch build that supports
    the GPU's compute capability (or run on CPU via `--asr-device cpu`).
    """

    def __init__(self, cfg: TranscribeConfig):
        self.cfg = cfg
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError as e:
                raise ImportError(
                    "nemo_toolkit not installed. `pip install -e '.[parakeet]'`"
                ) from e
            ref = self.cfg.model_path or self.cfg.model
            device = "cuda" if self.cfg.device in ("cuda", "auto") else "cpu"
            log.info("loading Parakeet (NeMo) model=%s device=%s", ref, device)
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=ref)
            model = model.to(device)
            if device == "cuda" and "float16" in (self.cfg.compute_type or ""):
                model = model.half()
            model.eval()
            self._model = model
        return self._model

    def transcribe(
        self, audio: np.ndarray, options: TranscribeOptions | None = None
    ) -> list[Word]:
        model = self._ensure_model()
        opts = options or TranscribeOptions()
        off = opts.time_offset_s
        samples = np.asarray(audio, dtype=np.float32)
        duration = len(samples) / SAMPLE_RATE

        # NeMo transcribe takes a list of audio file paths most reliably.
        with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
            _write_wav16(tf.name, samples)
            out = model.transcribe([tf.name], timestamps=True, verbose=False)

        if not out:
            return []
        hyp = out[0]
        text = (getattr(hyp, "text", None) or "").strip()
        if not text:
            return []

        words = _words_from_nemo_timestamps(getattr(hyp, "timestamp", None), off)
        if words:
            return words
        log.warning("Parakeet returned no parseable word timestamps; interpolating")
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


def _words_from_nemo_timestamps(timestamp, offset: float) -> list[Word]:
    """Map NeMo's `hypothesis.timestamp['word']` entries to Words. Each entry is a
    dict with 'word' + 'start'/'end' (seconds). Returns [] on any unexpected shape
    so the caller can fall back to interpolation."""
    if not isinstance(timestamp, dict):
        return []
    word_ts = timestamp.get("word")
    if not word_ts:
        return []
    out: list[Word] = []
    try:
        for e in word_ts:
            txt = str(e.get("word") or e.get("char") or "").strip()
            start = e.get("start")
            end = e.get("end")
            if start is None or end is None or not txt:
                continue
            out.append(Word(
                text=txt,
                start_s=float(start) + offset,
                end_s=float(end) + offset,
                probability=1.0,
            ))
    except (TypeError, ValueError, KeyError, AttributeError):
        return []
    return out


def _interpolate_words(text: str, offset: float, duration: float) -> list[Word]:
    """Spread word timings across [offset, offset+duration], weighted by token
    length — fallback when timestamps are unavailable."""
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
