from .base import Translator, TranslateOptions  # noqa: F401


def build_translator(cfg):
    """Construct a Translator from a TranslateConfig."""
    if cfg.backend == "transformers":
        from .gemma_transformers import GemmaTransformersTranslator
        return GemmaTransformersTranslator(cfg)
    if cfg.backend == "ollama":
        from .ollama_backend import OllamaTranslator
        return OllamaTranslator(cfg)
    if cfg.backend == "gemini":
        from .gemini_cloud import GeminiTranslator
        return GeminiTranslator(cfg)
    raise ValueError(f"Unknown translate backend: {cfg.backend!r}")
