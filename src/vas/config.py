from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict


# ---------- transcription ----------

class TranscribeConfig(BaseModel):
    """Whisper backend + model size + decoding params."""
    model_config = ConfigDict(extra="allow")

    name: str = "default"
    backend: Literal[
        "faster_whisper", "whisper_cpp", "trt_llm", "qwen3_asr", "openai_whisper",
        "parakeet", "canary_qwen", "granite_speech", "whisperx",
    ] = "faster_whisper"
    model: str = "large-v3-turbo"
    device: Literal["cuda", "cpu", "auto"] = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    best_of: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    language: str | None = None
    initial_prompt: str | None = None
    model_path: str | None = None  # ggml file / TRT engine dir


# ---------- translation ----------

class TranslateConfig(BaseModel):
    """Gemma-family translation model + runtime knobs."""
    model_config = ConfigDict(extra="allow")

    name: str = "default"
    backend: Literal["transformers", "ollama", "gemini"] = "transformers"
    model: str = "google/translategemma-12b-it"
    device: Literal["cuda", "cpu", "auto"] = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    quantization: Literal["none", "int8", "int4"] = "none"
    max_new_tokens: int = 512
    temperature: float = 0.0  # 0 -> greedy
    top_p: float = 1.0
    batch_size: int = 8

    # Source/target language. Defaults are placeholders; CLI can override.
    src_lang: str | None = None  # None -> assume already-detected by ASR
    tgt_lang: str = "en"

    # For ollama backend
    ollama_host: str = "http://127.0.0.1:11434"

    # For gemini backend (cloud)
    gemini_api_key_env: str = "GOOGLE_API_KEY"
    gemini_model: str = "gemini-2.5-flash"


# ---------- VAD ----------

class VadConfig(BaseModel):
    backend: Literal["silero"] = "silero"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 400
    speech_pad_ms: int = 200


# ---------- segmenter (Whisper-friendly ~30s windows) ----------

class SegmentConfig(BaseModel):
    target_seconds: float = Field(28.0, ge=5.0, le=60.0)
    min_seconds: float = Field(8.0, ge=1.0)
    max_seconds: float = Field(34.0, ge=10.0)


# ---------- subtitle cue assembly ----------

class CueConfig(BaseModel):
    max_chars_per_line: int = 42
    max_lines: int = 2
    max_duration_s: float = 6.0
    min_gap_ms: int = 80


# ---------- top-level ----------

class AppConfig(BaseModel):
    transcribe: TranscribeConfig = TranscribeConfig()
    translate: TranslateConfig | None = None
    vad: VadConfig = VadConfig()
    segment: SegmentConfig = SegmentConfig()
    cues: CueConfig = CueConfig()


# ---------- preset loaders ----------

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
TRANSCRIBE_DIR = CONFIGS_DIR / "transcribe"
TRANSLATE_DIR = CONFIGS_DIR / "translate"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_transcribe_preset(name_or_path: str) -> TranscribeConfig:
    p = Path(name_or_path)
    if not p.exists():
        p = TRANSCRIBE_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Transcribe preset {name_or_path!r} not found. "
            f"Looked in cwd and {TRANSCRIBE_DIR}."
        )
    return TranscribeConfig.model_validate(_load_yaml(p))


def load_translate_preset(name_or_path: str) -> TranslateConfig:
    p = Path(name_or_path)
    if not p.exists():
        p = TRANSLATE_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Translate preset {name_or_path!r} not found. "
            f"Looked in cwd and {TRANSLATE_DIR}."
        )
    return TranslateConfig.model_validate(_load_yaml(p))


def list_presets(kind: Literal["transcribe", "translate"]) -> list[str]:
    d = TRANSCRIBE_DIR if kind == "transcribe" else TRANSLATE_DIR
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))
