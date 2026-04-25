from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class TranslateOptions:
    src_lang: str | None = None
    tgt_lang: str = "en"
    glossary: dict[str, str] | None = None  # term -> preferred translation
    style: str | None = None  # e.g. "formal", "broadcast subtitle"


@runtime_checkable
class Translator(Protocol):
    """Translate a list of strings (one per cue) into target-language strings.

    Implementations should batch internally for throughput. Order is preserved
    1:1 -- empty inputs yield empty strings, never reordered.
    """

    def translate(
        self, texts: list[str], options: TranslateOptions | None = None
    ) -> list[str]: ...

    def close(self) -> None: ...


def build_prompt(text: str, src_lang: str | None, tgt_lang: str) -> str:
    """Generic Gemma instruction-tuned translation prompt.

    Used by backends that don't ship a model-specific chat template.
    """
    src = src_lang or "the source language"
    return (
        f"Translate the following subtitle text from {src} to {tgt_lang}. "
        f"Preserve meaning, tone, and any speaker punctuation. "
        f"Output ONLY the translation, no preamble, no quotes, no explanation.\n\n"
        f"Text: {text}"
    )
