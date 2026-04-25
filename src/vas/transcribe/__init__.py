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
    raise ValueError(f"Unknown transcribe backend: {cfg.backend!r}")
