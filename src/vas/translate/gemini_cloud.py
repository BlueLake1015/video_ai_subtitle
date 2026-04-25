from __future__ import annotations

import os

from ..config import TranslateConfig
from ..utils.logging import get_logger
from .base import TranslateOptions, build_prompt

log = get_logger(__name__)


class GeminiTranslator:
    """Cloud translation via Google Gemini API.

    Use when you don't want to host a Gemma model locally. Requires
    `pip install google-genai` and a key in $GOOGLE_API_KEY (or whichever
    env var is named in cfg.gemini_api_key_env).
    """

    def __init__(self, cfg: TranslateConfig):
        self.cfg = cfg
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from google import genai
        except ImportError as e:
            raise ImportError("`pip install google-genai`") from e
        api_key = os.environ.get(self.cfg.gemini_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"${self.cfg.gemini_api_key_env} is not set"
            )
        self._client = genai.Client(api_key=api_key)

    def translate(
        self, texts: list[str], options: TranslateOptions | None = None
    ) -> list[str]:
        if not texts:
            return []
        self._ensure_client()
        opts = options or TranslateOptions(
            src_lang=self.cfg.src_lang, tgt_lang=self.cfg.tgt_lang
        )
        outs: list[str] = []
        for t in texts:
            prompt = build_prompt(t, opts.src_lang, opts.tgt_lang)
            resp = self._client.models.generate_content(
                model=self.cfg.gemini_model,
                contents=prompt,
            )
            outs.append((getattr(resp, "text", "") or "").strip())
        return outs

    def close(self) -> None:
        self._client = None
