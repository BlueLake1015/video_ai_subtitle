from vas.config import (
    list_presets,
    load_transcribe_preset,
    load_translate_preset,
)


def test_all_transcribe_presets_load():
    names = list_presets("transcribe")
    assert names, "no transcribe presets discovered"
    for n in names:
        cfg = load_transcribe_preset(n)
        assert cfg.model
        assert cfg.backend in {
            "faster_whisper", "whisper_cpp", "trt_llm", "qwen3_asr", "openai_whisper",
            "parakeet", "canary_qwen", "granite_speech", "whisperx",
        }


def test_all_translate_presets_load():
    names = list_presets("translate")
    assert names, "no translate presets discovered"
    for n in names:
        cfg = load_translate_preset(n)
        assert cfg.model
        assert cfg.backend in {"transformers", "ollama", "gemini"}


def test_transcribe_preset_covers_size_range():
    """At minimum we expose tiny ... large quality."""
    names = set(list_presets("transcribe"))
    assert "tiny" in names
    assert "large-v3" in names or "quality" in names


def test_translate_preset_covers_size_range():
    names = set(list_presets("translate"))
    assert "edge" in names
    assert "balanced" in names or "quality" in names
