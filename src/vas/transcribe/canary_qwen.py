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


class CanaryQwenTranscriber:
    """NVIDIA Canary-Qwen-2.5B backend via NeMo (SALM speech-LLM).

    Requires the 'canary' extra:  pip install -e '.[canary]'

    FastConformer encoder + Qwen LLM decoder — tops the English Open ASR
    leaderboard on accuracy. **English only**, and the architecture produces no
    timestamps, so word timings are interpolated across each ~30 s VAD segment
    (the cue assembler still breaks readable cues; per-word timing is approximate).
    Trained max audio is 40 s, comfortably above our ~28 s segments. Runs on
    PyTorch — needs a torch build supporting the GPU's compute capability, or CPU.
    """

    def __init__(self, cfg: TranscribeConfig):
        self.cfg = cfg
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from nemo.collections.speechlm2.models import SALM
            except ImportError as e:
                raise ImportError(
                    "nemo_toolkit not installed. `pip install -e '.[canary]'`"
                ) from e
            ref = self.cfg.model_path or self.cfg.model
            device = "cuda" if self.cfg.device in ("cuda", "auto") else "cpu"
            log.info("loading Canary-Qwen (NeMo SALM) model=%s device=%s", ref, device)
            model = SALM.from_pretrained(ref)
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

        with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
            _write_wav16(tf.name, samples)
            prompt = f"Transcribe the following: {model.audio_locator_tag}"
            answer_ids = model.generate(
                prompts=[[{"role": "user", "content": prompt, "audio": [tf.name]}]],
                max_new_tokens=getattr(self.cfg, "max_new_tokens", 256),
            )
            text = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()

        if not text:
            return []
        # Canary-Qwen emits no timestamps; interpolate across the segment.
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
