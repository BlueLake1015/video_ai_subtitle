# offline_packages/

Tools and artifacts for installing **video_ai_subtitle** on an air-gapped machine. The directory has two roles depending on whether you're on the online build host or the offline target:

- **On an online machine** (must match the offline target's Ubuntu version + CPU architecture): the `build_*.sh` scripts populate `debs/`, `nvidia-debs/`, `pip-wheels/`, and `hf-cache/` from the network.
- **On the offline target**: `install.sh` consumes those artifacts and produces a working `.venv` with all models cached locally.

## Layout

```
offline_packages/
├── README.md                ← this file
├── build_packages.sh        ← gather .deb + .whl files (online machine)
├── build_models.sh          ← download Whisper + Gemma weights (online machine)
├── install.sh               ← install everything (offline target)
├── manifest.txt             ← inventory written at the end of each build
├── debs/                    ← generic OS packages (gitignored)
├── nvidia-debs/             ← NVIDIA driver only — separated because it requires a reboot
├── pip-wheels/              ← PyTorch + project deps as wheels, plus requirements.txt
└── hf-cache/                ← HuggingFace cache (Whisper + Gemma model weights)
```

The build artifacts (`debs/`, `nvidia-debs/`, `pip-wheels/`, `hf-cache/`, `manifest.txt`) are gitignored. The three `.sh` scripts and this README are committed.

## Why `nvidia-debs/` is separate

The NVIDIA driver `.debs` install kernel modules that need a reboot before the GPU is usable. The other `.debs` (ffmpeg, Python 3.11, X server, gcc, libllvm, etc.) install in place without rebooting.

`build_packages.sh` walks the apt-rdepends closure for `nvidia-driver-NNN-open` and **splits the result by name pattern**:

- Names matching `^(nvidia-|libnvidia-|linux-modules-nvidia-|linux-objects-nvidia-|linux-signatures-nvidia-|xserver-xorg-video-nvidia-)` go to `nvidia-debs/` (17 files, ~285 MB on Ubuntu 22.04)
- Everything else (gcc, libllvm15, xserver-xorg-core, libdrm, …) is folded into `debs/` along with ffmpeg/python deps so there's **no overlap** between the two directories.

`install.sh` honours the split: `nvidia-debs/` is applied first (and can be skipped via `SKIP_NVIDIA_DEBS=1` if the target already has a working driver), then `debs/`.

---

## Phase 1: build the bundle (online machine)

```bash
cd video_ai_subtitle

# 1. Gather .debs (~600 MB) and pip wheels (~3.9 GB)
bash offline_packages/build_packages.sh

# 2. Download Whisper + Gemma model weights (~6-100 GB depending on variants chosen)
#    Set HF_TOKEN first if any Gemma repo is gated for your HF account.
bash offline_packages/build_models.sh

# 3. Pack and transfer
tar czf bundle.tgz offline_packages/
scp bundle.tgz user@offline-host:~/video_ai_subtitle/
```

### `build_packages.sh` knobs

| Env var | Default | What it does |
|---|---|---|
| `TARGET_DRIVER` | `auto` (detects via nvidia-smi) | Pin to a specific driver major, e.g. `570`, `580` |
| `TARGET_KERNEL` | `auto` (build host's `uname -r`) | Kernel version whose `linux-headers` to gather (DKMS needs them on the offline target). Set to the target's `uname -r` if it differs from this host. Empty string skips. |
| `PYTHON_VERSION` | `3.11` | Python series to gather; must match `requires-python` in pyproject |
| `TORCH_CUDA` | `cu128` | PyTorch wheel index suffix; pick to match your driver's max-supported CUDA |
| `PIP_EXTRAS` | `translate,quant,dev` | Project extras to resolve (add `whispercpp,gemini` if needed) |
| `SKIP_DEBS` | `0` | Skip the apt phase (debs/ already populated) |
| `SKIP_NVIDIA` | `0` | Skip the nvidia-debs phase |
| `SKIP_PIP` | `0` | Skip the pip phase |
| `DRY_RUN` | `0` | Print every command instead of running |
| `OFFLINE_DIR` | `<repo>/offline_packages` | Override output location |

After both phases finish, the script writes `sha256sums.txt` covering every `.deb`, `.whl`, `.tar.gz`, `requirements.txt`, and `manifest.txt` in the bundle. `install.sh` verifies these on the target before doing anything else.

### `build_models.sh` knobs

| Env var | Default | What it does |
|---|---|---|
| `WHISPER_MODELS` | `tiny base small medium large-v3 distil-large-v3` | Whisper variants to grab (Systran/faster-whisper-* repos) |
| `WHISPER_TURBO_REPO` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Override the large-v3-turbo source repo |
| `GEMMA_MODELS` | `google/gemma-3-1b-it google/translategemma-4b-it google/translategemma-12b-it` | Gemma variants to grab; set empty to skip translation models |
| `OFFLINE_DIR` | `<repo>/offline_packages` | Override output location |

Models live in HuggingFace cache layout under `offline_packages/hf-cache/hub/models--<org>--<name>/...`, so `install.sh` can copy that directly into `~/.cache/huggingface/` on the target.

---

## Phase 2: install on the offline target

```bash
cd video_ai_subtitle
tar xzf bundle.tgz
bash offline_packages/install.sh
sudo reboot                    # only if a fresh NVIDIA driver was installed
```

### What `install.sh` does

0. **Integrity check** — `sha256sum -c sha256sums.txt` against every `.deb`/`.whl`/`.tar.gz`/`requirements.txt`/`manifest.txt`. Aborts on any mismatch (transfer corruption / tampering). Skip with `SKIP_VERIFY=1`.
1. **`nvidia-debs/`** — `apt-get install` from `nvidia-debs/*.deb` (driver kernel modules; needs reboot to take effect).
2. **`debs/`** — `apt-get install` from `debs/*.deb` (Python 3.11, ffmpeg, generic libraries, kernel headers if gathered).
3. **`.venv`** — `python3.11 -m venv` using the system Python from step 2.
4. **pip wheels** — `pip install --no-index --find-links pip-wheels/ -r pip-wheels/requirements.txt` (every dependency at the exact version frozen on the build host).
5. **HF cache** — copies `hf-cache/` into `~/.cache/huggingface/` (or whatever `HF_CACHE_TARGET` points at).
6. **`activate` patch** — appends `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to `.venv/bin/activate` so future runs don't try to phone home.

### `install.sh` knobs

| Env var | Default | What it does |
|---|---|---|
| `OFFLINE_DIR` | `<repo>/offline_packages` | Where to read the bundle from |
| `VENV_DIR` | `<repo>/.venv` | Where to create the venv |
| `PYTHON_VERSION` | `3.11` | Which python to call for `python -m venv` |
| `HF_CACHE_TARGET` | `~/.cache/huggingface` | Where the HF cache lands |
| `SKIP_VERIFY` | `0` | Don't run the sha256 integrity check |
| `SKIP_DEBS` | `0` | Don't install apt packages (already done) |
| `SKIP_NVIDIA_DEBS` | `0` | Don't install nvidia-debs (target already has driver) |
| `SKIP_PIP` | `0` | Don't install python wheels |
| `SKIP_MODELS` | `0` | Don't copy hf-cache |
| `DRY_RUN` | `0` | Preview, no changes |

---

## Bundle size reference

Order-of-magnitude on a typical Ubuntu 22.04 + RTX 4090/5090 build host:

| Component | Approx size |
|---|---|
| `debs/` (~400 .deb files) | 280–320 MB |
| `nvidia-debs/` (17 files, driver 570) | ~280 MB |
| `pip-wheels/` (torch+cu128 + 70 deps) | 3.9 GB |
| `hf-cache/` Whisper variants (tiny..large-v3 + turbo + distil) | ~6 GB |
| `hf-cache/` + Gemma 3 1B + TranslateGemma 4B | +12 GB |
| `hf-cache/` + TranslateGemma 12B (int8 useful preset) | +25 GB |
| `hf-cache/` + TranslateGemma 27B (only if you'll use `quality` preset) | +54 GB |

A minimal but useful bundle (transcribe-only, single Whisper model, no translation) clocks around **5 GB**. A full kit with TranslateGemma 12B is around **30 GB**.

---

## Pre-flight on the offline target

Before running `install.sh`, sanity-check the target environment:

```bash
. /etc/os-release && echo "OS:   $PRETTY_NAME"   # must match build host
uname -m                                          # must be x86_64
uname -r                                          # for the kernel-headers gap below
which sudo                                        # required by install.sh
df -h .                                           # ~5 GB free for venv + debs install
df -h ~                                           # +6-50 GB if hf-cache copies to ~/.cache
```

### Kernel-headers gap

`nvidia-dkms-XXX-open` compiles the kernel module against `linux-headers-$(uname -r)` at install time. That package is kernel-version-specific, so the `auto` default of `TARGET_KERNEL` (which gathers headers for the build host's kernel) only covers the offline target if both run the same kernel.

Three ways to handle this:

```bash
# Option A: build host runs the same kernel as the target -> default works
TARGET_KERNEL=auto bash offline_packages/build_packages.sh

# Option B: target runs a known different kernel -> pin it explicitly
TARGET_KERNEL=5.15.0-176-generic bash offline_packages/build_packages.sh

# Option C: have the target pre-install linux-headers-generic before going air-gapped
#   (this skips the issue entirely; `TARGET_KERNEL=` in the build is then fine)
sudo apt install -y linux-headers-generic     # on the still-online target
```

## Verifying the bundle before shipping

```bash
# What's in each artifact dir
cat offline_packages/manifest.txt | head -30

# Sanity-check sizes
du -sh offline_packages/{debs,nvidia-debs,pip-wheels,hf-cache} 2>/dev/null

# Make sure debs/ and nvidia-debs/ don't overlap
comm -12 \
    <(ls offline_packages/debs/      | sed 's/_[0-9].*//' | sort -u) \
    <(ls offline_packages/nvidia-debs/| sed 's/_[0-9].*//' | sort -u)
# (empty output = clean)

# Re-verify checksums locally before shipping
( cd offline_packages && sha256sum -c --quiet sha256sums.txt ) && echo OK
```

## See also

- [`../README.md`](../README.md) §1.B — full offline workflow narrative
- [`../README.md`](../README.md) §6 — common troubleshooting (driver mismatch, ctranslate2 cu13 issue, gated Gemma repos)
- [`../scripts/install_online.sh`](../scripts/install_online.sh) — equivalent installer for machines that *do* have internet access
