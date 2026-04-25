from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import numpy as np

from .audio.ffmpeg_args import build_args, _is_url, SAMPLE_RATE
from .audio.ffmpeg_source import FfmpegSource, SourceMode
from .audio.supervisor import SupervisedStreamSource
from .config import AppConfig
from .cues.assembler import CueAssembler
from .segment import Segment, WhisperSegmenter
from .transcribe import build_transcriber
from .transcribe.base import TranscribeOptions
from .translate import build_translator
from .translate.base import TranslateOptions
from .types import Cue, Word
from .utils.logging import get_logger
from .vad.silero import SileroBatchVAD
from .writers import writer_for

log = get_logger(__name__)


# ---------------- shared helpers ----------------

def _decode_full_audio(input_: str) -> np.ndarray:
    """Decode entire local file to a 16 kHz mono float32 array via ffmpeg."""
    argv = ["ffmpeg", *build_args(input_, mode="batch")]
    log.info("ffmpeg %s", " ".join(argv[1:]))
    proc = subprocess.run(argv, check=True, capture_output=True)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_segments(
    cfg: AppConfig, segments: list[Segment]
) -> list[Word]:
    transcriber = build_transcriber(cfg.transcribe)
    all_words: list[Word] = []
    for i, seg in enumerate(segments):
        opts = TranscribeOptions(
            language=cfg.transcribe.language,
            initial_prompt=cfg.transcribe.initial_prompt,
            time_offset_s=seg.start_s,
        )
        words = transcriber.transcribe(seg.audio, opts)
        all_words.extend(words)
        log.info(
            "segment %d/%d [%.2f-%.2f] (%.1fs) -> %d words",
            i + 1, len(segments), seg.start_s, seg.end_s, seg.duration_s, len(words),
        )
    return all_words


def _translate_cues(cfg: AppConfig, cues: list[Cue]) -> list[Cue]:
    if cfg.translate is None or not cues:
        return cues
    translator = build_translator(cfg.translate)
    opts = TranslateOptions(
        src_lang=cfg.translate.src_lang or cfg.transcribe.language,
        tgt_lang=cfg.translate.tgt_lang,
    )
    src_texts = [c.text.replace("\n", " ") for c in cues]
    log.info(
        "translating %d cues with %s -> %s",
        len(src_texts), cfg.translate.model, opts.tgt_lang,
    )
    translated = translator.translate(src_texts, opts)
    out: list[Cue] = []
    for c, new_text in zip(cues, translated):
        # Re-wrap the translated text using the same line-length budget.
        lines = _wrap_lines(new_text, cfg.cues.max_chars_per_line, cfg.cues.max_lines)
        out.append(Cue(
            index=c.index,
            start_s=c.start_s,
            end_s=c.end_s,
            lines=lines,
            words=c.words,  # keep source words for reference / karaoke
            speaker=c.speaker,
        ))
    return out


def _wrap_lines(text: str, max_chars: int, max_lines: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return [""]
    lines: list[str] = [""]
    for tok in tokens:
        candidate = f"{lines[-1]} {tok}".strip() if lines[-1] else tok
        if len(candidate) <= max_chars:
            lines[-1] = candidate
        else:
            if len(lines) >= max_lines:
                lines[-1] = candidate  # overflow last line rather than drop content
            else:
                lines.append(tok)
    return [ln for ln in lines if ln]


# ---------------- batch pipeline (local file) ----------------

def run_batch(cfg: AppConfig, input_: str, output: str) -> None:
    """Decode entire file, VAD-segment, transcribe, translate, write."""
    audio = _decode_full_audio(input_)
    log.info("decoded %.2fs of audio", len(audio) / SAMPLE_RATE)

    vad = SileroBatchVAD(cfg.vad)
    regions = vad.segment(audio, SAMPLE_RATE)
    log.info("VAD: %d speech regions", len(regions))

    segmenter = WhisperSegmenter(cfg.segment, cfg.vad)
    segments = segmenter.from_regions(audio, regions)
    log.info("Built %d Whisper segments (~%.0fs each)", len(segments), cfg.segment.target_seconds)

    words = _transcribe_segments(cfg, segments)

    cues = CueAssembler(cfg.cues).assemble(words)
    log.info("assembled %d cues", len(cues))

    cues = _translate_cues(cfg, cues)

    writer_for(output).write(cues, output)
    log.info("wrote %s", output)


# ---------------- live MPEG2-TS pipeline ----------------

async def run_live(cfg: AppConfig, input_: str, output: str) -> None:
    """Live ingest from MPEG2-TS URL: stream segments, append cues to file as they finalize."""
    source = (
        SupervisedStreamSource(input_)
        if _is_url(input_)
        else FfmpegSource(input_, mode=SourceMode.REALTIME)
    )

    transcriber = build_transcriber(cfg.transcribe)
    translator = build_translator(cfg.translate) if cfg.translate else None
    assembler = CueAssembler(cfg.cues)
    segmenter = WhisperSegmenter(cfg.segment, cfg.vad)
    writer = writer_for(output)

    # Live = append-as-we-go. We rebuild the file each flush so the writer's
    # full-render output stays consistent (SRT/TTML are not naturally append-only).
    cues_so_far: list[Cue] = []
    out_path = Path(output)

    loop = asyncio.get_event_loop()

    try:
        async for seg in segmenter.from_frames(source.frames()):
            opts = TranscribeOptions(
                language=cfg.transcribe.language,
                initial_prompt=cfg.transcribe.initial_prompt,
                time_offset_s=seg.start_s,
            )
            words = await loop.run_in_executor(
                None, transcriber.transcribe, seg.audio, opts,
            )
            new_cues = assembler.assemble(words)

            if translator is not None and new_cues:
                topts = TranslateOptions(
                    src_lang=cfg.translate.src_lang or cfg.transcribe.language,
                    tgt_lang=cfg.translate.tgt_lang,
                )
                src_texts = [c.text.replace("\n", " ") for c in new_cues]
                translated = await loop.run_in_executor(
                    None, translator.translate, src_texts, topts,
                )
                for c, t in zip(new_cues, translated):
                    c.lines = _wrap_lines(t, cfg.cues.max_chars_per_line, cfg.cues.max_lines)

            # Re-index cues onto the running list.
            for c in new_cues:
                c.index = len(cues_so_far) + 1
                cues_so_far.append(c)

            out_path.write_text(writer.render(cues_so_far), encoding="utf-8")
            log.info(
                "[live] seg [%.1f-%.1f] +%d cues -> %s (total %d)",
                seg.start_s, seg.end_s, len(new_cues), output, len(cues_so_far),
            )
    finally:
        await source.close()
