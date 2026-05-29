# video_ai_subtitle

Subtitle generation from video. Pipeline:

```
ffmpeg                    # local file or MPEG2-TS stream
  -> Silero VAD
  -> ~30s segmenter       # silence-aligned chunks Whisper likes
  -> ASR                  # faster-whisper (default) / whisper.cpp / TRT-LLM /
                          #   Qwen3-ASR / OpenAI Whisper / Parakeet / Canary-Qwen /
                          #   Granite-Speech / WhisperX  (see §5 ASR backends)
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

Two paths:

- **[1.A Online](#1a-online-install-recommended)** — single script, target machine has internet access.
- **[1.B Offline / air-gapped](#1b-offline-air-gapped-install)** — gather a bundle on an online machine, install on a target with no internet.

### 1.A Online install (recommended)

The whole bootstrap is one script. It installs the NVIDIA driver, Python 3.11, ffmpeg, creates `.venv`, and pip-installs the project with the right extras.

#### One-shot install

```bash
git clone <repo-url> video_ai_subtitle
cd video_ai_subtitle

bash scripts/install_online.sh
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
2. **NVIDIA driver** — installs `nvidia-driver-570` (override with `TARGET_DRIVER=…`). Skips when an equal-or-newer driver is already installed.
3. **ffmpeg** — `apt install ffmpeg` if missing.
4. **Python 3.11** — installs from deadsnakes PPA on 22.04 (24.04 has it in the main archive, with deadsnakes as fallback).
5. **venv** — creates `.venv` with `python3.11`. If an existing venv was built with a different Python version, it's removed and recreated.
6. **Project install** — `pip install -e ".[translate,quant,dev]"` inside the venv.
7. **Verification** — runs `nvidia-smi`; tells you if a reboot is required.

#### Override knobs

```bash
TARGET_DRIVER=580       bash scripts/install_online.sh   # use a specific driver major
PYTHON_VERSION=3.12     bash scripts/install_online.sh   # use Python 3.12 instead
PIP_EXTRAS=translate,quant,dev,whispercpp \
                        bash scripts/install_online.sh   # add the whisper.cpp backend
RECREATE_VENV=never     bash scripts/install_online.sh   # keep an existing venv as-is
SKIP_PYTHON=1           bash scripts/install_online.sh   # only do driver/ffmpeg
DRY_RUN=1               bash scripts/install_online.sh   # preview commands, no changes
```

#### Manual install (if you can't run the script)

```bash
# 1. NVIDIA driver (Ubuntu 22.04/24.04)
sudo apt update && sudo apt install -y nvidia-driver-570
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

#### Optional extras

| Extra | What it adds | Install |
|---|---|---|
| `translate` | Gemma via Hugging Face transformers (default backend) | always recommended |
| `quant`     | bitsandbytes for int8/int4 Gemma quantization on a 24 GB GPU | recommended |
| `whispercpp`| `pywhispercpp` for the whisper.cpp ASR backend | optional |
| `gemini`    | Cloud Gemini translation backend | optional |
| `dev`       | pytest, pytest-asyncio, ruff, **jiwer** (WER scoring) | for development |

Alternative ASR engines (each opt-in; see [§5 ASR backends](#asr-backends)):

| Extra | ASR backend it enables | Notes |
|---|---|---|
| `qwen` | `qwen3_asr` — Qwen3-ASR-1.7B | multilingual incl. zh/ja/ko; forced aligner |
| `openai-whisper` | `openai_whisper` — reference OpenAI Whisper | multilingual baseline |
| `parakeet` | `parakeet` — NVIDIA Parakeet-TDT (NeMo) | en + 24 EU; fastest; heavy NeMo dep |
| `canary` | `canary_qwen` — NVIDIA Canary-Qwen-2.5B (NeMo) | English only; heavy NeMo dep |
| `granite` | `granite_speech` — IBM Granite-Speech-4.1-2B | en/fr/de/es/pt/ja |
| `whisperx` | `whisperx` — faster-whisper + wav2vec2 alignment | self-contained; precise word timing |

Add the defaults via `pip install -e ".[translate,quant,whispercpp,gemini,dev]"` or
`make install-all` (which is `[all,dev]` — note `parakeet`/`canary`/`whisperx` are
**not** in `[all]` because of their heavy/conflicting deps; install those
individually, e.g. `pip install -e ".[qwen]"`).

#### Verify the install

```bash
source .venv/bin/activate
python --version           # Python 3.11.x
which vas                  # …/.venv/bin/vas
vas list-presets           # prints transcribe + translate preset names
nvidia-smi                 # GPU visible, driver version >= 555
ffmpeg -version | head -1  # ffmpeg >= 4.4
```

### 1.B Offline / air-gapped install

For machines with no internet access. Two phases:

**Phase 1 — on an online machine** (same Ubuntu version + same architecture as the offline target):

```bash
git clone <repo-url> video_ai_subtitle
cd video_ai_subtitle

# Step 1a: gather every .deb (apt packages, transitive) into offline_packages/debs
bash offline_packages/download_1_ubuntu_packages.sh

# Step 1b: gather every .whl/.tar.gz (pip) into offline_packages/pip-wheels
bash offline_packages/download_2_python_packages.sh

# Step 2: download Whisper + Gemma model weights into offline_packages/hf-cache/
#         Authenticate first — gated Gemma repos silently fetch only README/config
#         when the HF token is missing, leaving the snapshot suspiciously small.
hf auth login                 # or: export HF_TOKEN=hf_xxx
bash offline_packages/download_3_models.sh

# Step 3: split into 1 GiB parts under ../offline_packages_parts/ and ship
sudo apt install pv pigz       # progress bar + parallel gzip (much faster than vanilla gzip)
mkdir -p ../offline_packages_parts
tar cf - offline_packages/ \
  | pv -s $(du -sb offline_packages | cut -f1) \
  | pigz \
  | split -b 1G -d -a 3 - ../offline_packages_parts/bundle.tgz.

scp -r ../offline_packages_parts user@offline-host:~/
```

The build profile — target GPU, CUDA index, torch + Python versions — lives in one file, [`offline_packages/target.env`](offline_packages/target.env) (defaults to the RTX 4090 / sm_89 target), which all the scripts source; change the target there once, or override per-run on the CLI.

The download scripts are **idempotent**: rerun them after a partial transfer or when adding a new variant and they skip what's already on disk (apt packages by `apt-cache` size match; HF repos by completeness check). Pass `FORCE_REDOWNLOAD=1` to `download_3_models.sh` to bypass the model cache. The package gathering uses `apt-rdepends` for the transitive closure of `nvidia-driver-570`, `python3.11`, `ffmpeg`, etc., and pip wheels are resolved against the same `cu128` PyTorch index the online installer uses, then frozen into `pip-wheels/requirements.txt` for deterministic offline replay.

**Phase 2 — on the offline target** (assumes `~/video_ai_subtitle/` is checked out and `~/offline_packages_parts/` is its sibling, as scp'd above):

```bash
cd ~/video_ai_subtitle
cat ../offline_packages_parts/bundle.tgz.* | tar xzf -    # restores ./offline_packages/
# Run the three install steps IN ORDER (the numbered names are the order):
bash offline_packages/install_1_ubuntu_packages.sh        # apt: nvidia + generic debs
sudo reboot                                                # only if a kernel module .deb was installed
bash offline_packages/install_2_python_packages.sh        # venv + pip wheels
bash offline_packages/install_3_models.sh                 # copy model caches
```

The offline target needs `tar` and `gzip` (both ship with Ubuntu by default); `pv` and `pigz` are not required there.

The three install steps do:

1. **`install_1_ubuntu_packages.sh`** — `apt-get install` from the local `.deb` pool (nvidia-debs first, then generic; apt resolves order from the local files).
2. **`install_2_python_packages.sh`** — `python3.11 -m venv .venv` using the system Python from step 1, then `pip install --no-index --find-links offline_packages/pip-wheels -r requirements.txt`, then appends `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` to `.venv/bin/activate`.
3. **`install_3_models.sh`** — copies `hf-cache/` into `~/.cache/huggingface/` (and the optional `whisper-cache/` / `torch-hub-cache/`) so models are found locally.

#### Offline knobs

```bash
# Build phase (online):
TARGET_DRIVER=580           bash offline_packages/download_1_ubuntu_packages.sh   # apt: driver major
TORCH_CUDA=cu124            bash offline_packages/download_2_python_packages.sh   # pip: CUDA wheel index
PIP_EXTRAS=translate,quant,whispercpp,gemini,dev \
                            bash offline_packages/download_2_python_packages.sh   # pip: extras
# Default torch = newest cu128 wheel (correct for the RTX 4090 target). Pin for
# reproducible builds, or for an older GPU — e.g. a V100 (Volta) test box:
TORCH_CUDA=cu124 TORCH_VERSION=2.6.0 \
                            bash offline_packages/download_2_python_packages.sh

# Default WHISPER_MODELS covers all 7 transcribe presets (tiny / base / small /
# medium / large-v3 / distil-large-v3 / large-v3-turbo). Default GEMMA_MODELS
# covers all 5 HF-hosted translate presets:
#   google/gemma-3-1b-it, translategemma-{4b,12b,27b}-it, gemma-4-27b-it
# Trim either to skip variants you won't use:
WHISPER_MODELS="medium large-v3-turbo" \
GEMMA_MODELS="google/translategemma-4b-it" \
                            bash offline_packages/download_3_models.sh
GEMMA_MODELS=""             bash offline_packages/download_3_models.sh   # whisper only
FORCE_REDOWNLOAD=1          bash offline_packages/download_3_models.sh   # bypass HF skip-if-cached

# Alternative ASR engines are opt-in (default off). Enable per engine:
ASR_ENGINE_MODELS="Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-ForcedAligner-0.6B \
  nvidia/parakeet-tdt-0.6b-v3 nvidia/canary-qwen-2.5b \
  ibm-granite/granite-speech-4.1-2b" \
                            bash offline_packages/download_3_models.sh   # HF engines -> hf-cache/
OPENAI_WHISPER_MODELS="large-v3"   bash offline_packages/download_3_models.sh   # .pt -> whisper-cache/
WHISPERX_ALIGN_LANGS="en zh ja ko" bash offline_packages/download_3_models.sh   # align+VAD -> torch-hub-cache/

# Install phase (offline target) — run only the steps you need; each is its own script:
SKIP_NVIDIA_DEBS=1       bash offline_packages/install_1_ubuntu_packages.sh   # target already has a driver
HF_CACHE_TARGET=/data/hf bash offline_packages/install_3_models.sh            # custom cache location
DRY_RUN=1                bash offline_packages/install_2_python_packages.sh   # preview, no changes
```

#### Bundle size

Plan disk before running the build phase. **The default model list pulls every preset**, totaling ~160 GB. Trim aggressively if you only need a subset:

| Component | Size |
|---|---|
| `debs/` + `nvidia-debs/` (driver + python + ffmpeg + transitive) | ~600 MB |
| `pip-wheels/` (torch+cu128, transformers, faster-whisper, …) | ~4 GB |
| `hf-cache/` Whisper variants (tiny..large-v3 + turbo + distil) | ~10 GB |
| `hf-cache/` + Gemma 3 1B + TranslateGemma 4B | +12 GB |
| `hf-cache/` + TranslateGemma 12B | +25 GB |
| `hf-cache/` + TranslateGemma 27B | +54 GB |
| `hf-cache/` + Gemma 4 27B (flagship preset) | +54 GB |
| **Default total (all presets)** | **~160 GB** |
| Minimum useful (transcribe-only, single Whisper) | ~5 GB |
| Translate-capable (4B + Whisper turbo) | ~15 GB |

Trim with `WHISPER_MODELS` / `GEMMA_MODELS` env vars on the build script. The two 27B models (`translategemma-27b-it`, `gemma-4-27b-it`) are the biggest cost — drop them unless you're using the `quality` or `gemma4-flagship` preset.

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

The repo ships two pre-built fixtures under [`tests/fixtures/`](tests/fixtures/):
- `test_en_30s.mp4` — NASA Goddard SVS narrated explainer (public domain, English)
- `test_zh_30s.mp4` — Lu Xun *True Story of Ah Q* read by a LibriVox volunteer (public domain, Mandarin)

See [`tests/fixtures/README.md`](tests/fixtures/README.md) for source attribution, regeneration recipes, and ready-to-paste `vas subtitle …` commands for each fixture (en, en→ko, zh, zh→en, zh→ko).

`scripts/local_file_test.sh` runs the bundled English fixture through one or more transcribe + translate preset combinations (English → Korean by default), times each run, and prints a pass/fail summary table. Every `vas` invocation is **echoed before it runs**, so you can copy any failing command verbatim and re-run it for debugging.

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

- Outputs land in `/tmp/vas-test/` (override with `OUT_DIR=…`)
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

### 2.2 Long-form evaluation (WER + hallucination)

[`tests/data/`](tests/data/) holds >1-hour real-world clips for stress-testing the
full pipeline (gitignored except the small audio-only film + TED-LIUM set):

- A public-domain feature film (*Night of the Living Dead*) for **robustness** —
  does the pipeline survive 90+ min of real multi-speaker audio?
- A 5-talk **TED-LIUM** set with a verbatim reference transcript for **accuracy**.

[`tests/data/score_wer.py`](tests/data/score_wer.py) scores a transcript against
the reference (WER + a breakdown) **and flags ASR hallucinations** — repetition
loops, duplicate/scattered repeated cues, phantom phrases, and insertion clusters.
Run with `--ref` for WER, or omit `--ref` for hallucination-only (e.g. the film,
which has no ground truth):

```bash
vas subtitle tests/data/tedlium_5talks_en.m4a -o /tmp/ted.srt -t large-v3-turbo --src-lang en
python tests/data/score_wer.py --hyp /tmp/ted.srt --ref tests/data/tedlium_5talks_en_reference.txt
```

[`tests/data/README.md`](tests/data/README.md) documents provenance, regeneration,
and a **WER + speed benchmark of all the ASR engines** on the TED-LIUM set.

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

# English -> Korean (transcribe + translate)
vas subtitle input.mp4 -o out.srt \
    -t large-v3-turbo \
    -T balanced \
    --src-lang en --tgt-lang ko

# Chinese -> English
vas subtitle input.mp4 -o out.srt \
    -t large-v3-turbo \
    -T fast \
    --src-lang zh --tgt-lang en

# TTML output (IMSC1)
vas subtitle input.mp4 -o out.ttml -t large-v3-turbo

# Highest quality (slow)
vas subtitle input.mp4 -o out.srt -t quality -T quality --tgt-lang ja

# Try the bundled fixtures
vas subtitle tests/fixtures/test_en_30s.mp4 -o /tmp/en.srt -t medium --src-lang en
vas subtitle tests/fixtures/test_zh_30s.mp4 -o /tmp/zh.srt -t medium --src-lang zh
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

#### Local loopback test (no real network feed needed)

`scripts/stream_local_file.sh` re-streams any local media file as MPEG2-TS to a loopback address, so you can exercise the live pipeline end-to-end on one machine.

```bash
# Terminal A — start the receiver FIRST (UDP packets sent before the receiver
# is listening are silently dropped):
vas subtitle 'udp://127.0.0.1:5000' -o /tmp/live.ko.srt \
    -t large-v3-turbo -T fast \
    --src-lang en --tgt-lang ko --keep-source
# Writes both /tmp/live.ko.srt (Korean) and /tmp/live.en.srt (English source)
# incrementally. Drop -T / --tgt-lang / --keep-source if you only want ASR.

# Terminal B — start the streamer
bash scripts/stream_local_file.sh                                  # streams the bundled fixture
bash scripts/stream_local_file.sh tests/fixtures/test_en_30s.mp4   # explicit input
bash scripts/stream_local_file.sh --loop                           # loop forever
bash scripts/stream_local_file.sh --rtp                            # RTP-encapsulated TS (RFC 2250)
bash scripts/stream_local_file.sh --port 6000                      # different port
```

`/tmp/live.srt` will be rewritten roughly every ~28 s (one Whisper segment) with cumulative cues. For the 30 s test fixture you'll want `--loop` so the streamer keeps going long enough for at least one segment to finalize.

If you Ctrl+C the receiver and a leftover ffmpeg is still bound to the port, kill it manually: `pkill -f 'ffmpeg.*udp://'`. (vas now sets `PR_SET_PDEATHSIG=SIGTERM` on its ffmpeg child, so this only happens if ffmpeg was started by something else and orphaned.)

### 3.3 Keep both source and translated subtitles

When `--translate-preset` is set, the project translates each cue and writes only the target file. Pass `--keep-source` to additionally write the pre-translation cues to a sibling path:

```bash
# Writes BOTH out.ko.srt (Korean) AND out.en.srt (English source)
vas subtitle in.mp4 -o out.ko.srt \
    -t large-v3-turbo -T balanced \
    --src-lang en --tgt-lang ko --keep-source
```

Path-derivation rules for the source file:
- Output ends in `.{tgt}.{ext}` → swap tgt for src: `out.ko.srt` → `out.en.srt`
- Otherwise insert `.{src}` before the extension: `out.srt` → `out.en.srt`

For an explicit path or a different output format for the source, use `--src-output`:

```bash
# Korean as SRT, English source as TTML
vas subtitle in.mp4 -o out.ko.srt \
    -t large-v3-turbo -T balanced \
    --src-lang en --tgt-lang ko \
    --src-output out.en.ttml
```

`--src-output` implies `--keep-source` and works for live streams too — both files are rewritten incrementally as segments finalize.

### 3.4 Translate an existing subtitle file

```bash
vas translate-file in.srt -o out.srt -T quality --tgt-lang ja
```

Reads SRT or VTT (handles both `,` and `.` decimal separators), translates each cue with the chosen Gemma preset, writes the translated subs.

### 3.5 Make shortcuts

```bash
make run IN=input.mp4 OUT=out.srt TPRESET=large-v3-turbo
make run IN=input.mp4 OUT=out.srt TPRESET=large-v3-turbo MPRESET=balanced TGT=ko
make run-live IN=udp://239.0.0.1:5000 OUT=live.srt TPRESET=large-v3-turbo MPRESET=fast TGT=ko
```

### 3.6 Override knobs (CLI flags)

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
| `--keep-source` | also write pre-translation cues to a sibling path | `--keep-source` |
| `--src-output` | explicit path/format for source-language file (implies `--keep-source`) | `--src-output out.en.ttml` |
| `--log-level` | log verbosity | `--log-level DEBUG` |

### 3.7 Listing presets

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

| Preset | Model | Backend | Decoding | Notes |
|---|---|---|---|---|
| `tiny` | `tiny` | faster_whisper | greedy | fastest; quality limited |
| `base` | `base` | faster_whisper | greedy | |
| `small` | `small` | faster_whisper | beam=5 | |
| `medium` | `medium` | faster_whisper | beam=5 | |
| `large-v3` | `large-v3` | faster_whisper | beam=5 | full multilingual |
| `large-v3-turbo` | `large-v3-turbo` | faster_whisper | beam=5 | **default**, ~6× faster than v3 |
| `distil-large-v3` | `distil-large-v3` | faster_whisper | greedy | English-only, very fast |
| `quality` | `large-v3` | faster_whisper | beam=5, best_of=5, conditioning on | highest accuracy |
| `low-vram` | `medium` | faster_whisper | greedy, int8_float16 | for <8 GB GPUs |
| `qwen3-asr` | `Qwen/Qwen3-ASR-1.7B` | qwen3_asr | bf16 | non-Whisper engine; 52 langs (en/zh/ja/ko/…); forced aligner on by default; needs `[qwen]` extra |
| `openai-whisper` | `large-v3` | openai_whisper | beam=5 | reference OpenAI Whisper (PyTorch); needs `[openai-whisper]` extra |
| `parakeet` | `nvidia/parakeet-tdt-0.6b-v3` | parakeet | TDT | non-Whisper; English + 24 European langs (NO zh/ja/ko); needs `[parakeet]` extra (NeMo) |
| `canary-qwen` | `nvidia/canary-qwen-2.5b` | canary_qwen | LLM decode | non-Whisper speech-LLM; **English only**, no timestamps (interpolated); needs `[canary]` extra (NeMo) |
| `granite-speech` | `ibm-granite/granite-speech-4.1-2b` | granite_speech | LLM decode | non-Whisper speech-LLM; en/fr/de/es/pt/**ja** ASR (no zh/ko); no timestamps (interpolated); needs `[granite]` extra |
| `whisperx` | `Systran/faster-whisper-large-v3` | whisperx | self-contained | faster-whisper ASR + wav2vec2 forced alignment; runs its **own** VAD/batching/align (skips this project's VAD/segmenter); needs `[whisperx]` extra |

#### ASR backends

The transcribe backend is selected per preset (`backend:` field) or via
`--asr-backend`. Available: `faster_whisper` (default, CTranslate2), `whisper_cpp`
(needs `[whispercpp]`), `trt_llm`, `qwen3_asr` (needs `[qwen]`), `openai_whisper`
(needs `[openai-whisper]`), `parakeet` (needs `[parakeet]`), `canary_qwen`
(needs `[canary]`), `granite_speech` (needs `[granite]`), and `whisperx`
(needs `[whisperx]`).

`parakeet` is NVIDIA **Parakeet-TDT-0.6B-v3** via the NeMo toolkit — a
FastConformer-TDT model that tops the Open ASR leaderboard on English
speed/accuracy. Output is punctuated and capitalized with word-level timestamps.
It covers **English + 24 European languages** (auto-detected) but **not Chinese,
Japanese, or Korean** — so it's an English/European alternative, not a
replacement for the multilingual Whisper/Qwen engines. Install with `pip install
-e ".[parakeet]"`; the NeMo dependency is large, so it's intentionally **kept out
of the `[all]` extra**. Runs on PyTorch — same GPU/torch caveat below (or CPU).

`canary_qwen` is NVIDIA **Canary-Qwen-2.5B** via NeMo's `SALM` (a FastConformer
encoder + Qwen LLM decoder) — tops the Open ASR leaderboard on English *accuracy*.
**English only**, and the LLM-decoder architecture emits **no timestamps**, so word
timings are interpolated across each VAD segment (coarser cue boundaries). Heavier
than parakeet (2.5B). Shares the NeMo dependency; install with `pip install -e
".[canary]"` (also kept out of `[all]`).

`granite_speech` is IBM **Granite-Speech-4.1-2B** via HuggingFace transformers (a
conformer encoder + Granite LLM decoder). ASR for **English, French, German,
Spanish, Portuguese, and Japanese** (no Chinese/Korean ASR). Punctuated/capitalized
output; the base model emits **no timestamps** (interpolated — the `-plus` variant
adds them). It's transformers-based (no NeMo) and `granite_speech` is registered in
transformers ≥4.57. Install with `pip install -e ".[granite]"`.

`whisperx` is **WhisperX** — faster-whisper ASR + **wav2vec2 forced alignment** for
accurate word timestamps (alignment covers en/zh/ja/ko + ~40 langs). Unlike the
other backends it is **self-contained**: it runs its *own* Silero VAD, batched
transcription, and alignment over the whole file. The batch pipeline detects its
`consumes_full_audio` flag and **bypasses this project's VAD + segmenter** for it
(no double VAD); all other backends keep the normal VAD→segmenter path. Default ASR
model is the CT2 build `Systran/faster-whisper-large-v3`. Install with `pip install
-e ".[whisperx]"`.

> **Volta/older-GPU install note.** Recent WhisperX depends on `pyannote-audio>=4`,
> which forces `torch>=2.7` — that drops Volta (sm_70) support and would break the
> torch engines on such GPUs. On a box pinned to torch 2.6 (e.g. a V100), install
> WhisperX without disturbing the rest:
> `pip install --no-deps whisperx && pip install "pyannote.audio<4" nltk pandas`
> (with torch/torchaudio/ctranslate2/faster-whisper pinned to their current
> versions). On the documented target (RTX 4090+, torch cu128) the plain
> `pip install -e ".[whisperx]"` works as-is.

`openai_whisper` is the **reference OpenAI Whisper** implementation (the `whisper`
PyTorch package) — the same models as `faster_whisper` but OpenAI's own decoding
and word-timestamp alignment, useful as an accuracy baseline. Slower and heavier
than CTranslate2; language is an ISO code (or null to auto-detect). Install with
`pip install -e ".[openai-whisper]"`. Like `qwen3_asr` it runs on PyTorch, so the
GPU/torch caveat below applies (or run on CPU with `--asr-device cpu`).

`qwen3_asr` uses Alibaba's **Qwen3-ASR-1.7B** via the `qwen-asr` package — a
non-Whisper, transformer ASR. The base model returns plain (punctuated) text only,
so the `qwen3-asr` preset enables the `Qwen/Qwen3-ForcedAligner-0.6B` model by
default (`forced_aligner:` field) for **real word-level timestamps**; the backend
reattaches punctuation from the full transcript onto the aligned words. Set
`forced_aligner: null` to skip it (lighter — no extra ~0.6B model / VRAM — but
word timings are then interpolated across each VAD segment, giving coarser cue
boundaries). Install with `pip install -e ".[qwen]"`. **Note:** Qwen3-ASR runs on PyTorch, so it needs
a GPU whose compute capability your installed `torch` build supports — e.g. a
`torch` wheel built for the GPU's CC, or run on CPU (`--asr-device cpu`). The
`faster_whisper` backend (CTranslate2) is independent of the torch build.

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
bash scripts/install_online.sh   # rebuilds the venv with python3.11
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
TORCH_CUDA=cu124 bash scripts/install_online.sh
TORCH_CUDA=cpu   bash scripts/install_online.sh   # CPU-only build
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
  transcribe/      Transcriber protocol + faster-whisper / whisper.cpp / TRT-LLM / Qwen3-ASR / OpenAI Whisper / Parakeet / Canary-Qwen / Granite-Speech / WhisperX
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
  install_online.sh    one-shot driver + Python 3.11 + venv + project bootstrap (online)
  local_file_test.sh   end-to-end run against the bundled fixtures
  stream_local_file.sh re-streams a local file as MPEG2-TS to loopback for live testing
offline_packages/
  download_1_ubuntu_packages.sh  gathers .deb files (apt closure) into offline_packages/debs on an online machine
  download_2_python_packages.sh  gathers .whl/.tar.gz files into offline_packages/pip-wheels on an online machine
  download_3_models.sh      downloads Whisper + Gemma model variants into offline_packages/hf-cache/
  install_1_ubuntu_packages.sh  offline target: install nvidia + generic .debs (step 1)
  install_2_python_packages.sh  offline target: create venv + install pip wheels (step 2)
  install_3_models.sh           offline target: copy model caches into place (step 3)
  debs/, pip-wheels/, hf-cache/   populated by the build scripts
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
