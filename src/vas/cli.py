from __future__ import annotations

import asyncio
from typing import Optional

import typer

from .config import (
    AppConfig,
    list_presets,
    load_transcribe_preset,
    load_translate_preset,
)
from .utils.logging import setup_logging, get_logger

app = typer.Typer(add_completion=False, help="video_ai_subtitle CLI")
log = get_logger(__name__)


def _build_app_config(
    transcribe_preset: str,
    translate_preset: Optional[str],
    src_lang: Optional[str],
    tgt_lang: Optional[str],
    transcribe_overrides: dict,
    translate_overrides: dict,
) -> AppConfig:
    cfg = AppConfig()
    cfg.transcribe = load_transcribe_preset(transcribe_preset)
    if transcribe_overrides:
        data = cfg.transcribe.model_dump()
        data.update(transcribe_overrides)
        cfg.transcribe = type(cfg.transcribe).model_validate(data)
    if src_lang and not cfg.transcribe.language:
        cfg.transcribe.language = src_lang

    if translate_preset:
        cfg.translate = load_translate_preset(translate_preset)
        if translate_overrides:
            data = cfg.translate.model_dump()
            data.update(translate_overrides)
            cfg.translate = type(cfg.translate).model_validate(data)
        if src_lang:
            cfg.translate.src_lang = src_lang
        if tgt_lang:
            cfg.translate.tgt_lang = tgt_lang
    return cfg


def _is_live_url(input_: str) -> bool:
    from .audio.ffmpeg_args import _is_url
    return _is_url(input_)


@app.command("subtitle")
def subtitle_cmd(
    input_: str = typer.Argument(..., help="Local file path or MPEG2-TS URL (udp://, rtp://)"),
    output: str = typer.Option(..., "-o", "--output", help="Output path (.srt/.ttml/.vtt)"),
    transcribe_preset: str = typer.Option(
        "large-v3-turbo", "--transcribe-preset", "-t",
        help="Whisper preset (see `vas list-presets`)"),
    translate_preset: Optional[str] = typer.Option(
        None, "--translate-preset", "-T",
        help="Gemma preset (omit for transcription-only)"),
    src_lang: Optional[str] = typer.Option(None, "--src-lang"),
    tgt_lang: Optional[str] = typer.Option(None, "--tgt-lang"),
    # Transcription overrides
    asr_model: Optional[str] = typer.Option(None, "--asr-model"),
    asr_backend: Optional[str] = typer.Option(None, "--asr-backend",
        help="faster_whisper | whisper_cpp | trt_llm"),
    asr_device: Optional[str] = typer.Option(None, "--asr-device"),
    # Translation overrides
    mt_model: Optional[str] = typer.Option(None, "--mt-model"),
    mt_backend: Optional[str] = typer.Option(None, "--mt-backend",
        help="transformers | ollama | gemini"),
    mt_quant: Optional[str] = typer.Option(None, "--mt-quant",
        help="none | int8 | int4"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Generate subtitles from a local file or MPEG2-TS stream.

    Examples:

      vas subtitle in.mp4 -o out.srt -t large-v3-turbo
      vas subtitle in.mp4 -o out.srt -t large-v3 -T balanced --src-lang en --tgt-lang ko
      vas subtitle udp://239.0.0.1:5000 -o live.srt -t large-v3-turbo -T fast --tgt-lang ko
    """
    setup_logging(log_level)

    transcribe_overrides: dict = {}
    if asr_model: transcribe_overrides["model"] = asr_model
    if asr_backend: transcribe_overrides["backend"] = asr_backend
    if asr_device: transcribe_overrides["device"] = asr_device

    translate_overrides: dict = {}
    if mt_model: translate_overrides["model"] = mt_model
    if mt_backend: translate_overrides["backend"] = mt_backend
    if mt_quant: translate_overrides["quantization"] = mt_quant

    cfg = _build_app_config(
        transcribe_preset, translate_preset, src_lang, tgt_lang,
        transcribe_overrides, translate_overrides,
    )

    from .pipeline import run_batch, run_live
    if _is_live_url(input_):
        log.info("live ingest: %s", input_)
        asyncio.run(run_live(cfg, input_, output))
    else:
        log.info("batch: %s", input_)
        run_batch(cfg, input_, output)


@app.command("translate-file")
def translate_file_cmd(
    input_: str = typer.Argument(..., help="Existing .srt or .vtt file"),
    output: str = typer.Option(..., "-o", "--output"),
    translate_preset: str = typer.Option("balanced", "--translate-preset", "-T"),
    src_lang: Optional[str] = typer.Option(None, "--src-lang"),
    tgt_lang: str = typer.Option("en", "--tgt-lang"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Translate an existing subtitle file (.srt or .vtt) using a Gemma preset."""
    setup_logging(log_level)
    from .subtitle_io import read_cues
    from .translate import build_translator
    from .translate.base import TranslateOptions
    from .writers import writer_for

    cfg = load_translate_preset(translate_preset)
    if src_lang: cfg.src_lang = src_lang
    if tgt_lang: cfg.tgt_lang = tgt_lang

    cues = read_cues(input_)
    log.info("read %d cues from %s", len(cues), input_)

    translator = build_translator(cfg)
    src_texts = [c.text.replace("\n", " ") for c in cues]
    translated = translator.translate(
        src_texts,
        TranslateOptions(src_lang=cfg.src_lang, tgt_lang=cfg.tgt_lang),
    )
    for c, t in zip(cues, translated):
        c.lines = [t]  # caller can re-wrap if needed

    writer_for(output).write(cues, output)
    log.info("wrote %s", output)


@app.command("list-presets")
def list_presets_cmd():
    """Print available transcribe and translate presets."""
    print("Transcribe presets:")
    for n in list_presets("transcribe"):
        print(f"  {n}")
    print()
    print("Translate presets:")
    for n in list_presets("translate"):
        print(f"  {n}")


if __name__ == "__main__":
    app()
