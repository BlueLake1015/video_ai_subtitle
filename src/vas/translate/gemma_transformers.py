from __future__ import annotations

from typing import Iterable

from ..config import TranslateConfig
from ..utils.logging import get_logger
from .base import TranslateOptions, build_prompt

log = get_logger(__name__)


class GemmaTransformersTranslator:
    """In-process Gemma translation via Hugging Face transformers.

    Supports any Gemma-family model (Gemma 2/3/4, TranslateGemma) at any size.
    The model is loaded lazily on first translate() call. Quantization (int8 /
    int4 via bitsandbytes) is applied via load_kwargs so 12B+ models can fit
    on a 24 GB GPU alongside Whisper.
    """

    def __init__(self, cfg: TranslateConfig):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers + torch required. `pip install -e '.[translate]'`"
            ) from e

        log.info(
            "loading translation model=%s device=%s dtype=%s quant=%s",
            self.cfg.model, self.cfg.device, self.cfg.dtype, self.cfg.quantization,
        )

        load_kwargs: dict = {"trust_remote_code": False}
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        load_kwargs["torch_dtype"] = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        if self.cfg.quantization in ("int8", "int4"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ImportError(
                    "bitsandbytes required for int8/int4. "
                    "`pip install bitsandbytes`"
                ) from e
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=(self.cfg.quantization == "int8"),
                load_in_4bit=(self.cfg.quantization == "int4"),
                bnb_4bit_compute_dtype=load_kwargs["torch_dtype"],
            )
            # bnb manages device placement
            load_kwargs["device_map"] = "auto"
        else:
            if self.cfg.device == "cuda":
                load_kwargs["device_map"] = "cuda"
            elif self.cfg.device == "auto":
                load_kwargs["device_map"] = "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.model, **load_kwargs)
        self._model.eval()

    def translate(
        self, texts: list[str], options: TranslateOptions | None = None
    ) -> list[str]:
        if not texts:
            return []
        self._ensure_model()
        import torch

        opts = options or TranslateOptions(
            src_lang=self.cfg.src_lang, tgt_lang=self.cfg.tgt_lang
        )
        outs: list[str] = []
        bs = max(1, self.cfg.batch_size)

        for batch in _chunked(texts, bs):
            prompts = [
                self._format(t, opts.src_lang, opts.tgt_lang) for t in batch
            ]
            inputs = self._tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(self._model.device)

            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.temperature > 0,
                    temperature=self.cfg.temperature if self.cfg.temperature > 0 else 1.0,
                    top_p=self.cfg.top_p,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            for i, ids in enumerate(output):
                input_len = inputs["input_ids"][i].shape[0]
                gen = ids[input_len:]
                text = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
                outs.append(text)
        return outs

    def _format(self, text: str, src: str | None, tgt: str) -> str:
        """Use the model's own chat template when present; else generic prompt.

        TranslateGemma's chat template requires structured content:
            content = [{
                "type": "text",
                "source_lang_code": "en",
                "target_lang_code": "ko",
                "text": "...",
            }]
        Generic Gemma-instruct accepts a plain string message.
        """
        is_translategemma = "translategemma" in (self.cfg.model or "").lower()

        if is_translategemma:
            if not src:
                src = "en"  # template requires a source code; default to en
            messages = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": src,
                    "target_lang_code": tgt,
                    "text": text,
                }],
            }]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            messages = [{
                "role": "user",
                "content": build_prompt(text, src, tgt),
            }]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return build_prompt(text, src, tgt)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


def _chunked(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
