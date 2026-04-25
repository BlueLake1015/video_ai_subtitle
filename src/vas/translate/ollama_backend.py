from __future__ import annotations

import json
import urllib.request
import urllib.error

from ..config import TranslateConfig
from ..utils.logging import get_logger
from .base import TranslateOptions, build_prompt

log = get_logger(__name__)


class OllamaTranslator:
    """Translate via a local Ollama server.

    Easiest path to running quantized Gemma at any size:
        ollama pull gemma3:1b      # or 4b / 12b / 27b, gemma2:*, translategemma:*
        # Then reference via cfg.model = "gemma3:12b"

    Uses /api/generate with raw=False so Ollama applies the model's chat template.
    """

    def __init__(self, cfg: TranslateConfig):
        self.cfg = cfg

    def translate(
        self, texts: list[str], options: TranslateOptions | None = None
    ) -> list[str]:
        if not texts:
            return []
        opts = options or TranslateOptions(
            src_lang=self.cfg.src_lang, tgt_lang=self.cfg.tgt_lang
        )
        outs: list[str] = []
        for t in texts:
            prompt = build_prompt(t, opts.src_lang, opts.tgt_lang)
            outs.append(self._generate(prompt))
        return outs

    def _generate(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.max_new_tokens,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.cfg.ollama_host.rstrip('/')}/api/generate",
            data=body, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama request failed at {self.cfg.ollama_host}: {e}"
            ) from e
        return (data.get("response") or "").strip()

    def close(self) -> None:
        pass
