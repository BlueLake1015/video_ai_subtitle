# video_ai_subtitle

Subtitle generation from video. Pipeline:

```
ffmpeg                    # local file or MPEG2-TS stream
  -> Silero VAD
  -> ~30s segmenter       # silence-aligned chunks Whisper likes
  -> Whisper              # faster-whisper / whisper.cpp / TRT-LLM
  -> cue assembler
  -> Gemma translation    # optional (TranslateGemma / Gemma 3/4 / Gemini cloud)
  -> SRT / TTML / VTT
```

- **Input**: local video file *or* MPEG2-TS stream (`udp://`, `rtp://`)
- **Output**: SRT, TTML (IMSC1 text profile), WebVTT
- **Targets**: NVIDIA GPU (RTX 4090+ recommended); CPU works for small models

## Quick start

```bash
# System: ffmpeg >= 6.0 on PATH
python3.11 -m venv .venv && source .venv/bin/activate

pip install -e ".[translate,dev]"      # core + Gemma translation + dev
# or:  make install-all

# Transcribe-only
vas subtitle input.mp4 -o out.srt -t large-v3-turbo

# Transcribe + translate (Gemma)
vas subtitle input.mp4 -o out.srt \
    -t large-v3-turbo \
    -T balanced \
    --src-lang en --tgt-lang ko

# Live MPEG2-TS multicast -> SRT (file rewritten as cues finalize)
vas subtitle udp://239.0.0.1:5000 -o live.srt \
    -t large-v3-turbo -T fast --tgt-lang ko

# Translate an existing SRT
vas translate-file in.srt -o out.srt -T quality --tgt-lang ja

# List presets
vas list-presets
```

## Presets

### Transcription (`-t / --transcribe-preset`)

All Whisper model sizes, all swappable via flags. `compute_type=float16` is the default for CUDA; switch to `int8_float16` for memory-constrained boxes.

| Preset | Whisper model | Decoding | Notes |
|---|---|---|---|
| `tiny` | `tiny` | greedy | fastest; quality limited |
| `base` | `base` | greedy | |
| `small` | `small` | beam=5 | |
| `medium` | `medium` | beam=5 | |
| `large-v3` | `large-v3` | beam=5 | full multilingual |
| `large-v3-turbo` | `large-v3-turbo` | beam=5 | **default**, ~6x faster than v3 |
| `distil-large-v3` | `distil-large-v3` | greedy | English-only, very fast |
| `quality` | `large-v3` | beam=5, best_of=5, conditioning on | highest accuracy |
| `low-vram` | `medium` | greedy, int8_float16 | for <8 GB GPUs |

### Translation (`-T / --translate-preset`)

All Gemma sizes, all swappable. Quantization is set per-preset to fit on a 24 GB GPU alongside Whisper.

| Preset | Model | Backend | Quant | Approx VRAM |
|---|---|---|---|---|
| `edge` | `google/gemma-3-1b-it` | transformers | none | ~3 GB |
| `fast` | `google/translategemma-4b-it` | transformers | none | ~9 GB |
| `balanced` | `google/translategemma-12b-it` | transformers | int8 | ~13 GB |
| `quality` | `google/translategemma-27b-it` | transformers | int4 | ~16 GB |
| `gemma4-flagship` | `google/gemma-4-27b-it` | transformers | int4 | ~16 GB |
| `ollama-12b` | `gemma3:12b` (Ollama) | ollama | gguf | depends on Ollama tag |
| `cloud-gemini` | `gemini-2.5-flash` | gemini API | — | (cloud) |

Override any preset field with CLI flags: `--asr-model`, `--asr-backend`, `--asr-device`, `--mt-model`, `--mt-backend`, `--mt-quant`.

## CLI

```
vas subtitle INPUT -o OUTPUT [-t TRANSCRIBE_PRESET] [-T TRANSLATE_PRESET]
                              [--src-lang LANG] [--tgt-lang LANG]
                              [--asr-model X] [--asr-backend X] [--asr-device X]
                              [--mt-model X]  [--mt-backend X]  [--mt-quant X]

vas translate-file INPUT.srt -o OUTPUT.srt -T TRANSLATE_PRESET --tgt-lang LANG

vas list-presets
```

INPUT can be:
- a local file path (`.mp4`, `.mkv`, `.ts`, `.mov`, ...)
- `udp://host:port` for MPEG2-TS multicast
- `rtp://host:port` for RTP MPEG2-TS

Live ingest writes the SRT/TTML file incrementally as segments finalize; the file is fully valid at every moment, so external players can re-read it.

## Architecture

```
src/vas/
  audio/           ffmpeg subprocess + arg builders + supervisor (live reconnect)
  vad/             Silero (batch + streaming, ONNX)
  segment.py       VAD-aware ~30s Whisper-friendly chunker
  transcribe/      Transcriber protocol + faster-whisper / whisper.cpp / TRT-LLM
  translate/       Translator protocol + transformers / Ollama / Gemini backends
  cues/            word-timing -> cue assembler (linebreaks, duration, gap)
  writers/         SRT, TTML (IMSC1), VTT
  pipeline.py      run_batch(file) and run_live(MPEG2-TS) orchestrators
  subtitle_io.py   SRT/VTT reader for translate-file
  cli.py           Typer CLI
  config.py        Pydantic schema + preset loaders
configs/
  transcribe/*.yaml    transcription presets
  translate/*.yaml     translation presets
```

### Why separate VAD + ~30s segmenter?

Whisper was trained on 30-second windows; feeding it audio aligned to that boundary maximizes accuracy. We use VAD to find natural silence boundaries, then aggregate speech regions into ~28 s segments without ever cutting through a word. Compared to feeding Whisper raw 30-second audio slices: better word-boundary accuracy, better cue timing, and per-segment language/prompt control.

### Why per-cue translation instead of per-segment?

Cues are short, parallel-friendly (batch them through Gemma in groups), and align to the playback timeline so you don't have to re-segment after translation. Per-segment translation also produces cleaner Gemma outputs (less truncation, no timestamp tokens leaking into output).

### Live ingest detail

Live `udp://` / `rtp://` sources go through `SupervisedStreamSource`, which wraps an ffmpeg subprocess with exponential-backoff reconnect. Frames flow into `WhisperSegmenter.from_frames` (streaming Silero VAD), which yields ~28 s segments at silence boundaries. Each segment is transcribed and (optionally) translated; the SRT/TTML file is rewritten after each segment so any external player picks up new cues automatically.

For UDP TS specifically, ffmpeg gets `-fifo_size 1000000 -overrun_nonfatal 1` so brief network bursts don't kill the demuxer.

## Backends

### faster-whisper (default)

CTranslate2-backed; best Whisper performance on NVIDIA GPUs. `compute_type=float16` for 4090, `int8_float16` for tighter memory.

### whisper.cpp

```bash
pip install -e ".[whispercpp]"
vas subtitle in.mp4 -o out.srt -t large-v3-turbo --asr-backend whisper_cpp
```

Uses `pywhispercpp` (in-process libwhisper bindings). Build a CUDA whisper.cpp via `make whispercpp-build` and point `transcribe.model_path` at a ggml-*.bin file for native CUDA.

### TensorRT-LLM (stub)

`src/vas/transcribe/trt_llm.py` defines the interface; build a TRT-LLM Whisper engine following NVIDIA's `examples/whisper`, then wire its runtime into `_load_engine()`. Expected ~2x throughput vs. faster-whisper on RTX 4090.

### Gemma translation backends

- **`transformers`** (default): in-process, supports any Gemma model (Gemma 2/3/4, TranslateGemma) at any size, with optional bitsandbytes int8/int4 quantization.
- **`ollama`**: routes through a local Ollama server. Easiest path to running quantized GGUFs of any Gemma size — `ollama pull gemma3:12b`, then set `cfg.model: gemma3:12b` and `backend: ollama`.
- **`gemini`**: cloud Gemini API. Set `$GOOGLE_API_KEY`. Use when you don't want local hosting overhead.

All three implement the same `Translator` protocol (`src/vas/translate/base.py`), so swapping is a config change.

## Development

```bash
make test           # pytest (no GPU needed for the suite)
make lint           # ruff check
make fmt            # ruff format
```

Tests cover pure-Python logic: ffmpeg argv builders, segmenter aggregation, cue assembly, writers, SRT round-trip, time formatters, and config/preset loading. Integration tests that load real models (Whisper / Gemma) should go in `tests/integration/` behind an env-var gate.

## License

MIT.
