from .base import Transcriber, TranscribeOptions  # noqa: F401


def build_transcriber(cfg):
    """Construct a Transcriber from a TranscribeConfig."""
    if cfg.backend == "faster_whisper":
        from .faster_whisper import FasterWhisperTranscriber
        return FasterWhisperTranscriber(cfg)
    if cfg.backend == "whisper_cpp":
        from .whisper_cpp import WhisperCppTranscriber
        return WhisperCppTranscriber(cfg)
    if cfg.backend == "trt_llm":
        from .trt_llm import TrtLlmTranscriber
        return TrtLlmTranscriber(cfg)
    if cfg.backend == "qwen3_asr":
        from .qwen3_asr import Qwen3AsrTranscriber
        return Qwen3AsrTranscriber(cfg)
    if cfg.backend == "openai_whisper":
        from .openai_whisper import OpenAiWhisperTranscriber
        return OpenAiWhisperTranscriber(cfg)
    if cfg.backend == "parakeet":
        from .parakeet import ParakeetTranscriber
        return ParakeetTranscriber(cfg)
    if cfg.backend == "canary_qwen":
        from .canary_qwen import CanaryQwenTranscriber
        return CanaryQwenTranscriber(cfg)
    if cfg.backend == "granite_speech":
        from .granite_speech import GraniteSpeechTranscriber
        return GraniteSpeechTranscriber(cfg)
    if cfg.backend == "whisperx":
        from .whisperx_engine import WhisperXTranscriber
        return WhisperXTranscriber(cfg)
    raise ValueError(f"Unknown transcribe backend: {cfg.backend!r}")
