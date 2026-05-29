# offline_packages/

Tools and artifacts for installing **video_ai_subtitle** on an air-gapped machine. The directory has two roles depending on whether you're on the online build host or the offline target:

- **On an online machine** (must match the offline target's Ubuntu version + CPU architecture): the `download_*.sh` scripts populate `debs/`, `nvidia-debs/`, `pip-wheels/`, and `hf-cache/` from the network.
- **On the offline target**: the `install_[123]_*.sh` scripts (run in order) consume those artifacts and produce a working `.venv` with all models cached locally.

## Layout

```
offline_packages/
├── README.md                     ← this file
├── target.env                    ← build profile: target GPU / CUDA / torch / python
├── download_1_ubuntu_packages.sh   ← gather .deb files / apt closure (online machine)
├── download_2_python_packages.sh   ← gather .whl/.tar.gz pip wheels (online machine)
├── download_3_models.sh            ← download Whisper + Gemma weights (online machine)
├── install_1_ubuntu_packages.sh  ← offline target: nvidia + generic .debs (step 1)
├── install_2_python_packages.sh  ← offline target: venv + pip wheels (step 2)
├── install_3_models.sh           ← offline target: copy model caches (step 3)
├── manifest_ubuntu.txt           ← inventory of the apt artifacts
├── manifest_python.txt           ← inventory of the pip artifacts
├── debs/                         ← generic OS packages (gitignored)
├── nvidia-debs/                  ← NVIDIA driver only — separated because it requires a reboot
├── pip-wheels/                   ← PyTorch + project deps as wheels, plus requirements.txt
├── hf-cache/                     ← HuggingFace cache (Whisper + Gemma + opt-in ASR engines)
├── whisper-cache/                ← openai-whisper .pt weights (only if OPENAI_WHISPER_MODELS set)
└── torch-hub-cache/              ← whisperx align + Silero VAD (only if WHISPERX_ALIGN_LANGS set)
```

The build artifacts (`debs/`, `nvidia-debs/`, `pip-wheels/`, `hf-cache/`, `manifest_*.txt`) are gitignored. The six `.sh` scripts (3 `download_*` + 3 `install_[123]_*`), `target.env`, and this README are committed.

## Build profile: `target.env`

`target.env` is the **single source of truth** for what the bundle targets — GPU class, CUDA index, torch, and Python versions. All six scripts source it on startup, so you change the target in one place:

```bash
: "${TARGET_DRIVER:=570}"      # nvidia driver major (.deb)
: "${PYTHON_VERSION:=3.11}"    # python series
: "${TORCH_CUDA:=cu128}"       # PyTorch wheel index (CUDA 12.8)
: "${TORCH_VERSION:=2.11.0}"   # pinned torch+torchaudio (reproducible)
```

Default profile = **NVIDIA RTX 4090 (Ada / sm_89)**, Ubuntu 22.04/24.04. Every value is assign-if-unset, so an explicit `VAR=... bash <script>` on the command line overrides the file. To build for a different target without editing the file, point at another profile: `TARGET_ENV=/path/to/v100.env bash …`, or override on the CLI (e.g. a Volta/sm_70 V100 test box: `TORCH_CUDA=cu124 TORCH_VERSION=2.6.0 …`).

## Why `nvidia-debs/` is separate

The NVIDIA driver `.debs` install kernel modules that need a reboot before the GPU is usable. The other `.debs` (ffmpeg, Python 3.11, X server, gcc, libllvm, etc.) install in place without rebooting.

`download_1_ubuntu_packages.sh` walks the apt-rdepends closure for `nvidia-driver-NNN-open` and **splits the result by name pattern**:

- Names matching `^(nvidia-|libnvidia-|linux-modules-nvidia-|linux-objects-nvidia-|linux-signatures-nvidia-|xserver-xorg-video-nvidia-)` go to `nvidia-debs/` (17 files, ~285 MB on Ubuntu 22.04)
- Everything else (gcc, libllvm15, xserver-xorg-core, libdrm, …) is folded into `debs/` along with ffmpeg/python deps so there's **no overlap** between the two directories.

`install_1_ubuntu_packages.sh` honours the split: `nvidia-debs/` is applied first (and can be skipped via `SKIP_NVIDIA_DEBS=1` if the target already has a working driver), then `debs/`.

---

## Phase 1: build the bundle (online machine)

```bash
cd video_ai_subtitle

# 1a. Gather .debs (~600 MB). Idempotent — re-runs skip files already on disk
#     with matching apt-cache size.
bash offline_packages/download_1_ubuntu_packages.sh
# 1b. Gather pip wheels (~3.9 GB).
bash offline_packages/download_2_python_packages.sh

# 2. Download Whisper + Gemma weights (~10-160 GB depending on variants chosen).
#    Authenticate FIRST. Without auth, gated Gemma repos silently fetch only
#    README/config (a 44 KB snapshot that looks "complete" but isn't usable).
hf auth login                  # or: export HF_TOKEN=hf_xxx
bash offline_packages/download_3_models.sh

# 3. Pack into 1 GiB parts and transfer.
#    pigz = parallel gzip (5-10× faster than vanilla gzip on multi-core hosts);
#    output stays gzip-compatible so the offline target needs no special tools.
sudo apt install pv pigz
mkdir -p ../offline_packages_parts
tar cf - offline_packages/ \
  | pv -s $(du -sb offline_packages | cut -f1) \
  | pigz \
  | split -b 1G -d -a 3 - ../offline_packages_parts/bundle.tgz.

scp -r ../offline_packages_parts user@offline-host:~/
```

All three download scripts are **idempotent and safe to re-run**:

- `download_1_ubuntu_packages.sh` looks up each package's expected filename and size via `apt-get download --print-uris`, skips files already on disk at the right size, and replaces stale (truncated/corrupt) ones.
- `download_2_python_packages.sh` validates wheels with a zip/tar integrity check before `pip download` runs, so partially-downloaded files from interrupted builds get re-fetched instead of silently shipped.
- `download_3_models.sh` checks each HF repo for `refs/main`, complete snapshots, no `*.incomplete` blobs, and at least one ≥ 10 MiB blob (the latter rejects "anonymous-fetch-only-README" snapshots when auth is missing). Pass `FORCE_REDOWNLOAD=1` to bypass.

### `download_1_ubuntu_packages.sh` knobs (apt)

| Env var | Default | What it does |
|---|---|---|
| `TARGET_DRIVER` | `auto` (detects via nvidia-smi) | Pin to a specific driver major, e.g. `570`, `580` |
| `TARGET_KERNEL` | `auto` (build host's `uname -r`) | Kernel version whose `linux-headers` to gather (DKMS needs them on the offline target). Set to the target's `uname -r` if it differs from this host. Empty string skips. |
| `PYTHON_VERSION` | `3.11` | Python series whose `.deb`s to gather; must match `requires-python` in pyproject |
| `SKIP_DEBS` | `0` | Skip the generic apt phase (debs/ already populated) |
| `SKIP_NVIDIA` | `0` | Skip the nvidia-debs phase |
| `DRY_RUN` | `0` | Print every command instead of running |
| `OFFLINE_DIR` | `<repo>/offline_packages` | Override output location |

### `download_2_python_packages.sh` knobs (pip)

| Env var | Default | What it does |
|---|---|---|
| `PYTHON_VERSION` | `3.11` | Python series to resolve wheels against |
| `TORCH_CUDA` | `cu128` | PyTorch wheel index suffix; pick to match your driver's max-supported CUDA |
| `TORCH_VERSION` | *(latest)* | Pin torch+torchaudio. Default takes the newest wheel (correct for the RTX 4090 / sm_89 target). Set for reproducible builds, or for an older GPU whose latest wheel dropped support — e.g. a V100 (Volta/sm_70) needs `TORCH_CUDA=cu124 TORCH_VERSION=2.6.0`. |
| `PIP_EXTRAS` | `translate,quant,dev` | Project extras to resolve (add `whispercpp,gemini,qwen,…` if needed) |
| `DRY_RUN` | `0` | Print every command instead of running |
| `OFFLINE_DIR` | `<repo>/offline_packages` | Override output location |

Each script writes its own inventory (`manifest_ubuntu.txt` / `manifest_python.txt`).

To bundle wheels for **every** ASR/translate engine, set `PIP_EXTRAS` to all groups
(the `target.env` torch pin handles the rest):

```bash
PIP_EXTRAS="translate,quant,whispercpp,gemini,dev,qwen,openai-whisper,parakeet,canary,granite,whisperx" \
  bash offline_packages/download_2_python_packages.sh
```

This resolves cleanly against the default RTX 4090 profile (`torch 2.11.0+cu128`) —
verified to land a consistent set (whisperx 3.7.2, nemo-toolkit 2.7.3,
transformers 4.57.6, ctranslate2 4.6.3). It pulls **many GB** of wheels (NeMo +
whisperx + alignment trees), so expect a long download. The matching model weights
are a separate opt-in in [`download_3_models.sh`](#download_3_modelssh-knobs)
(`ASR_ENGINE_MODELS` / `OPENAI_WHISPER_MODELS` / `WHISPERX_ALIGN_LANGS`).

### `download_3_models.sh` knobs

| Env var | Default | What it does |
|---|---|---|
| `WHISPER_MODELS` | `tiny base small medium large-v3 distil-large-v3` | Whisper variants to grab. Names map to `Systran/faster-whisper-*` except `distil-*` which use `Systran/faster-distil-whisper-*` (different word order). |
| `WHISPER_TURBO_REPO` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Override the large-v3-turbo source repo |
| `GEMMA_MODELS` | `google/gemma-3-1b-it google/translategemma-4b-it google/translategemma-12b-it google/translategemma-27b-it google/gemma-4-27b-it` | Gemma variants to grab; covers every HF-hosted translate preset. Set empty to skip translation models. |
| `ASR_ENGINE_MODELS` | *(empty)* | **Opt-in.** Space-separated HF repos for the alternative ASR engines → `hf-cache/`. Full set: `Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-ForcedAligner-0.6B nvidia/parakeet-tdt-0.6b-v3 nvidia/canary-qwen-2.5b ibm-granite/granite-speech-4.1-2b` |
| `OPENAI_WHISPER_MODELS` | *(empty)* | **Opt-in.** openai-whisper model names (e.g. `large-v3`) → `whisper-cache/` (.pt from OpenAI's CDN). Needs the `[openai-whisper]` extra in `$VENV`. |
| `WHISPERX_ALIGN_LANGS` | *(empty)* | **Opt-in.** Language codes (e.g. `en zh ja ko`) to pre-warm whisperx alignment + Silero VAD → `torch-hub-cache/` (+ non-en align models to `hf-cache/`). Needs the `[whisperx]` extra in `$VENV`. |
| `FORCE_REDOWNLOAD` | `0` | Skip the per-repo completeness check and re-fetch every model |
| `OFFLINE_DIR` | `<repo>/offline_packages` | Override output location |

**Auth requirement.** Run `hf auth login` (or export `HF_TOKEN`) **before** calling this script. The script only sets `HF_HUB_CACHE` (the cache location) and leaves `HF_HOME` at its default, so the user-level token stored in `~/.cache/huggingface/token` is found. The script logs the token source on startup; if it warns "no HF_TOKEN env var and no token at …", gated Gemma repos will fetch only public files (44 KB README/config) and look "cached" on the next run until you authenticate and re-fetch.

Models live in HuggingFace cache layout under `offline_packages/hf-cache/hub/models--<org>--<name>/...`, so `install_3_models.sh` can copy that directly into `~/.cache/huggingface/` on the target.

---

## Phase 2: install on the offline target

Layout assumption: `~/video_ai_subtitle/` is checked out and `~/offline_packages_parts/` is its sibling (matching what `scp -r` produced in Phase 1 step 3). `tar` and `gzip` are needed (both ship with Ubuntu by default); `pv` and `pigz` are not required on the target.

Run the three install scripts **in order** — the numbered names are the order:

```bash
cd ~/video_ai_subtitle
cat ../offline_packages_parts/bundle.tgz.* | tar xzf -      # restores ./offline_packages/
bash offline_packages/install_1_ubuntu_packages.sh          # apt: nvidia + generic debs
sudo reboot                                                  # only if a fresh NVIDIA driver was installed
bash offline_packages/install_2_python_packages.sh          # venv + pip wheels
bash offline_packages/install_3_models.sh                   # copy model caches
```

Add `v` to the tar flags (`tar xzvf -`) if you want filename-by-filename progress output during extraction.

### What the install steps do

**`install_1_ubuntu_packages.sh`**
1. **`nvidia-debs/`** — `apt-get install` from `nvidia-debs/*.deb` (driver kernel modules; needs reboot to take effect).
2. **`debs/`** — `apt-get install` from `debs/*.deb` (Python 3.11, ffmpeg, generic libraries, kernel headers if gathered).

**`install_2_python_packages.sh`**
3. **`.venv`** — `python3.11 -m venv` using the system Python from step 1.
4. **pip wheels** — `pip install --no-index --find-links pip-wheels/ -r pip-wheels/requirements.txt` (every dependency at the exact version frozen on the build host), then the project itself.
5. **`activate` patch** — appends `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to `.venv/bin/activate` so future runs don't try to phone home.

**`install_3_models.sh`**
6. **HF cache** — copies `hf-cache/` into `~/.cache/huggingface/` (or whatever `HF_CACHE_TARGET` points at). If present, also copies `whisper-cache/` → `~/.cache/whisper` (openai-whisper) and `torch-hub-cache/` → `~/.cache/torch` (whisperx align + VAD).

### install-script knobs

Each script reads only the knobs it needs; running a script *is* the choice (no `SKIP_DEBS`/`SKIP_PIP`/`SKIP_MODELS` — just don't run that step).

| Env var | Script | Default | What it does |
|---|---|---|---|
| `OFFLINE_DIR` | all | `<repo>/offline_packages` | Where to read the bundle from |
| `SKIP_NVIDIA_DEBS` | install_1 | `0` | Don't install nvidia-debs (target already has a driver) |
| `VENV_DIR` | install_2 | `<repo>/.venv` | Where to create the venv |
| `PYTHON_VERSION` | install_2 | `3.11` (from target.env) | Which python to call for `python -m venv` |
| `HF_CACHE_TARGET` | install_3 | `~/.cache/huggingface` | Where the HF cache lands |
| `WHISPER_CACHE_TARGET` | install_3 | `~/.cache/whisper` | Where the openai-whisper `.pt` cache lands (if bundled) |
| `TORCH_HOME_TARGET` | install_3 | `~/.cache/torch` | Where the whisperx torch.hub cache lands (if bundled) |
| `DRY_RUN` | all | `0` | Preview, no changes |

---

## Bundle size reference

Order-of-magnitude on a typical Ubuntu 22.04 + RTX 4090/5090 build host:

| Component | Approx size |
|---|---|
| `debs/` (~400 .deb files) | 280–320 MB |
| `nvidia-debs/` (17 files, driver 570) | ~280 MB |
| `pip-wheels/` (torch+cu128 + 70 deps) | 3.9 GB |
| `hf-cache/` Whisper variants (tiny..large-v3 + turbo + distil) | ~10 GB |
| `hf-cache/` + Gemma 3 1B + TranslateGemma 4B | +12 GB |
| `hf-cache/` + TranslateGemma 12B (int8 — `balanced` preset) | +25 GB |
| `hf-cache/` + TranslateGemma 27B (int4 — `quality` preset) | +54 GB |
| `hf-cache/` + Gemma 4 27B (`gemma4-flagship` preset) | +54 GB |
| **Default total (all presets)** | **~160 GB** |

A minimal but useful bundle (transcribe-only, single Whisper model, no translation) clocks around **5 GB**. A translate-capable kit with the 4B model and Whisper turbo is around **15 GB**. The two 27B variants are the dominant cost — `GEMMA_MODELS="google/gemma-3-1b-it google/translategemma-4b-it google/translategemma-12b-it"` drops the bundle to ~50 GB while still covering `edge`, `fast`, and `balanced` translate presets.

---

## Pre-flight on the offline target

Before running the install scripts, sanity-check the target environment:

```bash
. /etc/os-release && echo "OS:   $PRETTY_NAME"   # must match build host
uname -m                                          # must be x86_64
uname -r                                          # for the kernel-headers gap below
which sudo                                        # required by install_1_ubuntu_packages.sh
df -h .                                           # ~5 GB free for venv + debs install
df -h ~                                           # +6-50 GB if hf-cache copies to ~/.cache
```

### Kernel-headers gap

`nvidia-dkms-XXX-open` compiles the kernel module against `linux-headers-$(uname -r)` at install time. That package is kernel-version-specific, so the `auto` default of `TARGET_KERNEL` (which gathers headers for the build host's kernel) only covers the offline target if both run the same kernel.

Three ways to handle this:

```bash
# Option A: build host runs the same kernel as the target -> default works
TARGET_KERNEL=auto bash offline_packages/download_1_ubuntu_packages.sh

# Option B: target runs a known different kernel -> pin it explicitly
TARGET_KERNEL=5.15.0-176-generic bash offline_packages/download_1_ubuntu_packages.sh

# Option C: have the target pre-install linux-headers-generic before going air-gapped
#   (this skips the issue entirely; `TARGET_KERNEL=` in the build is then fine)
sudo apt install -y linux-headers-generic     # on the still-online target
```

## Verifying the bundle before shipping

```bash
# What's in each artifact dir
cat offline_packages/manifest_ubuntu.txt offline_packages/manifest_python.txt | head -40

# Sanity-check sizes
du -sh offline_packages/{debs,nvidia-debs,pip-wheels,hf-cache} 2>/dev/null

# Make sure debs/ and nvidia-debs/ don't overlap
comm -12 \
    <(ls offline_packages/debs/      | sed 's/_[0-9].*//' | sort -u) \
    <(ls offline_packages/nvidia-debs/| sed 's/_[0-9].*//' | sort -u)
# (empty output = clean)
```

## See also

- [`../README.md`](../README.md) §1.B — full offline workflow narrative
- [`../README.md`](../README.md) §6 — common troubleshooting (driver mismatch, ctranslate2 cu13 issue, gated Gemma repos)
- [`../scripts/install_online.sh`](../scripts/install_online.sh) — equivalent installer for machines that *do* have internet access
