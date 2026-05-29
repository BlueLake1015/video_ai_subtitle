from __future__ import annotations

import numpy as np

from ..config import TranscribeConfig
from ..types import Word
from ..utils.logging import get_logger
from .base import TranscribeOptions

log = get_logger(__name__)

SAMPLE_RATE = 16000

_DTYPE_NAME = {
    "float16": "float16", "bfloat16": "bfloat16", "float32": "float32",
    "int8_float16": "float16", "int8": "float16",
}

# Granite-Speech transcribes whatever language is spoken; the instruction is in
# English. ASR is supported for en / fr / de / es / pt / ja.
_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."


class GraniteSpeechTranscriber:
    """IBM Granite-Speech backend (speech-LLM) via HuggingFace transformers.

    Requires the 'granite' extra:  pip install -e '.[granite]'

    A conformer speech encoder + Granite LLM decoder. ASR for English, French,
    German, Spanish, Portuguese, and Japanese (no Chinese/Korean ASR). Output is
    punctuated and capitalized. The base model emits no timestamps, so word timings
    are interpolated across each ~30 s VAD segment (the `-plus` variant adds real
    timestamps). Runs on PyTorch — needs a torch build supporting the GPU's compute
    capability, or CPU.
    """

    def __init__(self, cfg: TranscribeConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._device = "cpu"

    def _ensure_model(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            except ImportError as e:
                raise ImportError(
                    "transformers not installed. `pip install -e '.[granite]'`"
                ) from e
            ref = self.cfg.model_path or self.cfg.model
            self._device = "cuda" if self.cfg.device in ("cuda", "auto") else "cpu"
            dtype = getattr(torch, _DTYPE_NAME.get(self.cfg.compute_type, "bfloat16"))
            log.info(
                "loading Granite-Speech model=%s device=%s dtype=%s",
                ref, self._device, dtype,
            )
            self._processor = AutoProcessor.from_pretrained(ref)
            self._tokenizer = self._processor.tokenizer
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                ref, device_map=self._device, dtype=dtype
            )
            self._model.eval()
        return self._model

    def transcribe(
        self, audio: np.ndarray, options: TranscribeOptions | None = None
    ) -> list[Word]:
        import torch

        model = self._ensure_model()
        opts = options or TranscribeOptions()
        off = opts.time_offset_s
        samples = np.asarray(audio, dtype=np.float32)
        duration = len(samples) / SAMPLE_RATE

        wav = torch.from_numpy(samples).unsqueeze(0)  # [1, N] mono
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": _PROMPT}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(prompt, wav, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=getattr(self.cfg, "max_new_tokens", 256),
                do_sample=False,
                num_beams=max(1, self.cfg.beam_size or 1),
            )
        n_in = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, n_in:].unsqueeze(0)
        text = self._tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )[0].strip()

        if not text:
            return []
        return _interpolate_words(text, off, duration)

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None


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
