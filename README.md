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
- **Python**: 3.11 (required; auto-installed by the bootstrap script)

---

## 1. Installation

The whole bootstrap is one script. It installs the NVIDIA driver, Python 3.11, ffmpeg, creates `.venv`, and pip-installs the project with the right extras.

### One-shot install (recommended)

```bash
git clone <repo-url> video_ai_subtitle
cd video_ai_subtitle

bash scripts/install_profile_a.sh
# Run as your user. The script will sudo only for apt commands.
# DO NOT prefix with sudo unless you have to: it auto-de-elevates if you do.
```

If a kernel-module install happened (driver was new or upgraded), reboot before continuing:

```bash
sudo reboot
# After reboot:
nvidia-smi                # should print a table with the GPU and driver version
```

What the script does:

1. **OS / GPU preflight** — checks Ubuntu 22.04 / 24.04 and detects the NVIDIA GPU via `lspci`.
2. **NVIDIA driver** — installs `nvidia-driver-560` (override with `TARGET_DRIVER=…`). Skips when an equal-or-newer driver is already installed.
3. **ffmpeg** — `apt install ffmpeg` if missing.
4. **Python 3.11** — installs from deadsnakes PPA on 22.04 (24.04 has it in the main archive, with deadsnakes as fallback).
5. **venv** — creates `.venv` with `python3.11`. If an existing venv was built with a different Python version, it's removed and recreated.
6. **Project install** — `pip install -e ".[translate,quant,dev]"` inside the venv.
7. **Verification** — runs `nvidia-smi`; tells you if a reboot is required.

### Override knobs

```bash
TARGET_DRIVER=580       bash scripts/install_profile_a.sh   # use a specific driver major
PYTHON_VERSION=3.12     bash scripts/install_profile_a.sh   # use Python 3.12 instead
PIP_EXTRAS=translate,quant,dev,whispercpp \
                        bash scripts/install_profile_a.sh   # add the whisper.cpp backend
RECREATE_VENV=never     bash scripts/install_profile_a.sh   # keep an existing venv as-is
SKIP_PYTHON=1           bash scripts/install_profile_a.sh   # only do driver/ffmpeg
DRY_RUN=1               bash scripts/install_profile_a.sh   # preview commands, no changes
```

### Manual install (if you can't run the script)

```bash
# 1. NVIDIA driver (Ubuntu 22.04/24.04)
sudo apt update && sudo apt install -y nvidia-driver-560
sudo reboot

# 2. Python 3.11 (Ubuntu 22.04 only; 24.04 ships it)
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 3. ffmpeg
sudo apt install -y ffmpeg

# 4. venv + project
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e ".[translate,quant,dev]"
```

### Optional extras

| Extra | What it adds | Install |
|---|---|---|
| `translate` | Gemma via Hugging Face transformers (default backend) | always recommended |
| `quant`     | bitsandbytes for int8/int4 Gemma quantization on a 24 GB GPU | recommended |
| `whispercpp`| `pywhispercpp` for the whisper.cpp ASR backend | optional |
| `gemini`    | Cloud Gemini translation backend | optional |
| `dev`       | pytest, pytest-asyncio, ruff | for development |

Add via `pip install -e ".[translate,quant,whispercpp,gemini,dev]"` or `make install-all`.

### Verify the install

```bash
source .venv/bin/activate
python --version           # Python 3.11.x
which vas                  # …/.venv/bin/vas
vas list-presets           # prints transcribe + translate preset names
nvidia-smi                 # GPU visible, driver version >= 555
ffmpeg -version | head -1  # ffmpeg >= 4.4
```

---

## 2. Testing

All commands assume `.venv` is active **or** that you use `make` (which uses the venv automatically).

```bash
make test                  # full pytest suite, ~0.2s, no GPU/model required
make lint                  # ruff check
make fmt                   # ruff format
```

Or directly:

```bash
.venv/bin/python -m pytest -q
```

Tests cover pure-Python logic only: ffmpeg argv builders, segmenter aggregation, cue assembly, writers, SRT round-trip, time formatters, config/preset loading, translation prompt builder. They don't load Whisper or Gemma models — those need a GPU and would be slow.

### 2.1 End-to-end test against a local file

`scripts/local_file_test.sh` runs the bundled fixture (`tests/fixtures/test_en_30s.mp4`) through one or more transcribe + translate preset combinations (English → Korean by default), times each run, and prints a pass/fail summary table. Every `vas` invocation is **echoed before it runs**, so you can copy any failing command verbatim and re-run it for debugging.

```bash
# Default: 1 transcribe (medium) + 1 translate (medium + fast). Fast.
bash scripts/local_file_test.sh

# Full matrix: every transcribe size + several translate combos.
bash scripts/local_file_test.sh --full
```

What it covers:

| Mode | Transcribe presets | Translate presets (transcribe → translate) |
|---|---|---|
| default | `medium` | `medium` → `fast` |
| `--full` | `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `distil-large-v3`, `quality` | `tiny`+`edge`, `medium`+`fast`, `large-v3-turbo`+`balanced` |

- Outputs land in `/tmp/vas-smoke/` (override with `OUT_DIR=…`)
- Each case asserts the output exists and contains ≥ 1 cue
- Translation cases are auto-skipped when no `HF_TOKEN` / `huggingface-cli login` is found
- Final exit code is non-zero if any case failed

Override knobs:

```bash
FIXTURE=path/to/your.mp4   bash scripts/local_file_test.sh           # different input
SRC_LANG=ja TGT_LANG=en    bash scripts/local_file_test.sh --full    # different language pair
OUT_DIR=/tmp/foo           bash scripts/local_file_test.sh           # different output dir
KEEP_OUTPUTS=0             bash scripts/local_file_test.sh           # wipe outputs before each run
```

`bash scripts/local_file_test.sh --help` prints the usage block at the top of the script.

For HF token / Gemma license setup (required to enable translation cases), see section 6 (Troubleshooting).

---

## 3. Running

### 3.1 Local file → subtitle file (batch)

```bash
# Activate venv first (or use `make run …`)
source .venv/bin/activate

# Transcribe-only
vas subtitle input.mp4 -o out.srt -t large-v3-turbo

# Specific source language (skips language detection, faster + more accurate)
vas subtitle input.mp4 -o out.srt -t large-v3-turbo --src-lang en

# Transcribe + translate to Korean
vas subtitle input.mp4 -o out.srt \
    -t large-v3-turbo \
    -T balanced \
    --src-lang en --tgt-lang ko

# TTML output (IMSC1)
vas subtitle input.mp4 -o out.ttml -t large-v3-turbo

# Highest quality (slow)
vas subtitle input.mp4 -o out.srt -t quality -T quality --tgt-lang ja
```

### 3.2 MPEG2-TS live stream → subtitle file (incremental)

The output file is rewritten after each segment finalizes, so any subtitle-aware player picks up new cues automatically.

```bash
# UDP MPEG2-TS multicast
vas subtitle udp://239.0.0.1:5000 -o live.srt \
    -t large-v3-turbo -T fast --tgt-lang ko

# RTP MPEG2-TS unicast
vas subtitle rtp://0.0.0.0:5000 -o live.srt -t large-v3-turbo

# Local .ts file paced to wall-clock (simulates live)
vas subtitle stream.ts -o live.srt -t large-v3-turbo --src-lang en
```

The pipeline auto-detects URL schemes and applies low-latency / TS-buffer ffmpeg flags. Network drops trigger exponential-backoff reconnect.

### 3.3 Translate an existing subtitle file

```bash
vas translate-file in.srt -o out.srt -T quality --tgt-lang ja
```

Reads SRT or VTT (handles both `,` and `.` decimal separators), translates each cue with the chosen Gemma preset, writes the translated subs.

### 3.4 Make shortcuts

```bash
make run IN=input.mp4 OUT=out.srt TPRESET=large-v3-turbo
make run IN=input.mp4 OUT=out.srt TPRESET=large-v3-turbo MPRESET=balanced TGT=ko
make run-live IN=udp://239.0.0.1:5000 OUT=live.srt TPRESET=large-v3-turbo MPRESET=fast TGT=ko
```

### 3.5 Override knobs (CLI flags)

Any preset field can be overridden inline:

| Flag | Affects | Example |
|---|---|---|
| `--asr-model` | transcribe model | `--asr-model distil-large-v3` |
| `--asr-backend` | transcribe backend | `--asr-backend whisper_cpp` |
| `--asr-device` | transcribe device | `--asr-device cpu` |
| `--mt-model` | translate model | `--mt-model google/gemma-4-27b-it` |
| `--mt-backend` | translate backend | `--mt-backend ollama` |
| `--mt-quant` | translate quantization | `--mt-quant int4` |
| `--src-lang` / `--tgt-lang` | language pair | `--src-lang en --tgt-lang ko` |
| `--log-level` | log verbosity | `--log-level DEBUG` |

### 3.6 Listing presets

```bash
vas list-presets
```

Prints all transcribe + translate preset names (the YAML files under `configs/`).

---

## 4. Makefile (optional shortcuts)

The `Makefile` is a thin task runner — every target is a named alias for a longer command, and **every target depends on `.venv`**, so the right Python and dependencies are always used. You don't have to `source .venv/bin/activate` first.

It is **strictly optional**: you can ignore it and run `vas …` / `pytest` / `pip` directly inside the venv. It just saves typing for the most common operations.

```bash
make help                  # list every target with a description
```

Common targets:

| Target | Equivalent direct command |
|---|---|
| `make venv` | create `.venv` with `python3.11`, upgrade pip/wheel/setuptools |
| `make install` | `.venv/bin/pip install -e .` |
| `make install-translate` | `.venv/bin/pip install -e ".[translate]"` |
| `make install-all` | `.venv/bin/pip install -e ".[all,dev]"` |
| `make test` | `.venv/bin/python -m pytest -q` |
| `make lint` | `.venv/bin/python -m ruff check src tests` |
| `make fmt` | `.venv/bin/python -m ruff format src tests` |
| `make list-presets` | `.venv/bin/python -m vas list-presets` |
| `make run IN=in.mp4 OUT=out.srt TPRESET=large-v3-turbo` | `vas subtitle in.mp4 -o out.srt -t large-v3-turbo` |
| `make run-live IN=udp://239.0.0.1:5000 OUT=live.srt TPRESET=large-v3-turbo MPRESET=fast TGT=ko` | live ingest with translation |
| `make models-asr ASR_MODEL=large-v3-turbo` | pre-warm the faster-whisper model cache |
| `make whispercpp-build` | clone whisper.cpp submodule and build with CUDA |
| `make clean` | remove caches and build artifacts |

Override knobs via env vars:

```bash
PYTHON_VERSION=3.12 make venv          # use a different Python
VENV=.venv-prod make install           # use a different venv path
```

If you prefer not to use `make`, every target's underlying command is shown in the table above — run those directly.

---

## 5. Presets

### Transcription (`-t / --transcribe-preset`)

| Preset | Model | Decoding | Notes |
|---|---|---|---|
| `tiny` | `tiny` | greedy | fastest; quality limited |
| `base` | `base` | greedy | |
| `small` | `small` | beam=5 | |
| `medium` | `medium` | beam=5 | |
| `large-v3` | `large-v3` | beam=5 | full multilingual |
| `large-v3-turbo` | `large-v3-turbo` | beam=5 | **default**, ~6× faster than v3 |
| `distil-large-v3` | `distil-large-v3` | greedy | English-only, very fast |
| `quality` | `large-v3` | beam=5, best_of=5, conditioning on | highest accuracy |
| `low-vram` | `medium` | greedy, int8_float16 | for <8 GB GPUs |

### Translation (`-T / --translate-preset`)

| Preset | Model | Backend | Quant | Approx VRAM |
|---|---|---|---|---|
| `edge` | `google/gemma-3-1b-it` | transformers | none | ~3 GB |
| `fast` | `google/translategemma-4b-it` | transformers | none | ~9 GB |
| `balanced` | `google/translategemma-12b-it` | transformers | int8 | ~13 GB |
| `quality` | `google/translategemma-27b-it` | transformers | int4 | ~16 GB |
| `gemma4-flagship` | `google/gemma-4-27b-it` | transformers | int4 | ~16 GB |
| `ollama-12b` | `gemma3:12b` (Ollama) | ollama | gguf | depends on Ollama tag |
| `cloud-gemini` | `gemini-2.5-flash` | gemini API | — | (cloud) |

To author a custom preset, drop a YAML file in `configs/transcribe/` or `configs/translate/`; pass its stem (or path) to `-t` / `-T`.

---

## 6. Troubleshooting

### `nvidia-smi has failed because it couldn't communicate with the NVIDIA driver`

Reboot. The driver kernel module needs the kernel to be running with it loaded; this happens after a reboot following a fresh install or upgrade.

```bash
sudo reboot
nvidia-smi      # should now print the GPU table
```

### `ERROR: Package 'video-ai-subtitle' requires a different Python: 3.10.x not in '>=3.11'`

Your venv was built with Python 3.10. Recreate it with 3.11:

```bash
sudo rm -rf .venv               # if any files inside are root-owned
bash scripts/install_profile_a.sh   # rebuilds the venv with python3.11
```

### `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`

Your driver supports CUDA 12.8 but the installed PyTorch was built against a newer CUDA (e.g. cu130). The default PyPI `torch` wheel is whatever the latest CUDA build is — which may exceed what your driver supports.

The install script auto-detects the driver's max-supported CUDA via `nvidia-smi` and pins the matching wheel index (`cu128` / `cu124` / `cu121` / `cu118` / `cpu`). If you bypassed it or the detection went wrong, fix manually:

```bash
# Find the right index for your driver
nvidia-smi | grep "CUDA Version"     # e.g. "CUDA Version: 12.8" -> use cu128

# Reinstall torch + torchaudio from that index
.venv/bin/pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
```

Override the auto-detection on the install script:

```bash
TORCH_CUDA=cu124 bash scripts/install_profile_a.sh
TORCH_CUDA=cpu   bash scripts/install_profile_a.sh   # CPU-only build
```

### `OSError: libcudart.so.13: cannot open shared object file`

CTranslate2 4.7+ ships CUDA 13 wheels which need `libcudart.so.13`; on driver-570 (max CUDA 12.8) those won't load. The project pins `ctranslate2 >=4.5,<4.7` to avoid this. If you somehow ended up with 4.7+:

```bash
.venv/bin/pip install --upgrade 'ctranslate2>=4.5,<4.7'
```

### `TemplateError: User role must provide content as an iterable with exactly one item ...`

That's TranslateGemma's chat template rejecting a free-form prompt. The `transformers` translation backend handles this automatically (sends structured `source_lang_code`/`target_lang_code`/`text`). If you see this from a custom call site, mimic the pattern in [`src/vas/translate/gemma_transformers.py`](src/vas/translate/gemma_transformers.py).

### whisper.cpp backend fails to load model

By default `pywhispercpp` uses CPU. For CUDA, build whisper.cpp from source:

```bash
make whispercpp-build       # clones submodule, builds with -DGGML_CUDA=1
```

Then point at a ggml-*.bin via `--mt-model`/`model_path` in your transcribe preset.

### Gemma int4 quantization fails

Install the `quant` extra:

```bash
.venv/bin/pip install -e ".[quant]"
```

This pulls `bitsandbytes`. Some CUDA versions need a specific bnb build — see https://github.com/TimDettmers/bitsandbytes for the matrix.

### Live UDP stream loses audio frames

Usual cause: kernel UDP receive buffer too small. The script already passes `-fifo_size 1000000 -overrun_nonfatal 1` to ffmpeg. For very high-bitrate streams, also raise the kernel buffer:

```bash
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.rmem_default=16777216
```

---

## 7. Architecture

```
src/vas/
  audio/           ffmpeg subprocess + arg builders + live-stream supervisor
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
scripts/
  install_profile_a.sh   one-shot driver + Python 3.11 + venv + project bootstrap
```

### Why separate VAD + ~30s segmenter?

Whisper was trained on 30-second windows; aligning input to that boundary maximizes accuracy. VAD finds natural silence boundaries, then the segmenter aggregates speech regions into ~28 s chunks without ever cutting through a word. Result: better word-boundary timing, cleaner cues, per-segment language/prompt control.

### Why per-cue translation instead of per-segment?

Cues are short, parallel-friendly (batchable through Gemma), and align to the playback timeline so you don't re-segment after translation. Per-cue prompts also produce cleaner Gemma outputs (no timestamp tokens leaking into translation).

### Live ingest detail

`udp://` / `rtp://` sources go through `SupervisedStreamSource`, which wraps the ffmpeg subprocess with exponential-backoff reconnect. Frames flow into `WhisperSegmenter.from_frames` (streaming Silero VAD), which yields ~28 s segments at silence boundaries. Each segment is transcribed and (optionally) translated; the SRT/TTML file is rewritten after each segment so any external player picks up new cues automatically.

For UDP TS specifically, ffmpeg gets `-fifo_size 1000000 -overrun_nonfatal 1` so brief network bursts don't kill the demuxer.

---

## 8. License

MIT.
