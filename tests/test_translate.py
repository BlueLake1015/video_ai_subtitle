from vas.config import TranslateConfig
from vas.translate.base import build_prompt


def test_build_prompt_contains_languages_and_text():
    p = build_prompt("hello world", "en", "ko")
    assert "en" in p and "ko" in p
    assert "hello world" in p
    assert "ONLY the translation" in p


def test_build_prompt_handles_unknown_src_lang():
    p = build_prompt("hi", None, "ja")
    assert "source language" in p.lower() or "ja" in p


def test_translate_config_defaults_are_sane():
    cfg = TranslateConfig()
    assert cfg.backend == "transformers"
    assert cfg.tgt_lang == "en"
    assert cfg.batch_size > 0


class _FakeTranslator:
    """Stand-in implementing the Translator protocol for pipeline-level tests."""
    def __init__(self, suffix=" [tr]"):
        self.suffix = suffix
        self.calls = 0
    def translate(self, texts, options=None):
        self.calls += 1
        return [t + self.suffix for t in texts]
    def close(self):
        pass


def test_fake_translator_satisfies_protocol():
    from vas.translate.base import Translator
    t = _FakeTranslator()
    assert isinstance(t, Translator)
    assert t.translate(["a", "b"]) == ["a [tr]", "b [tr]"]
