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

# 1. Gather .debs (~600 MB) and pip wheels (~3.9 GB).
#    Idempotent — re-runs skip files already on disk with matching apt-cache size.
bash offline_packages/build_packages.sh

# 2. Download Whisper + Gemma weights (~10-160 GB depending on variants chosen).
#    Authenticate FIRST. Without auth, gated Gemma repos silently fetch only
#    README/config (a 44 KB snapshot that looks "complete" but isn't usable).
hf auth login                  # or: export HF_TOKEN=hf_xxx
bash offline_packages/build_models.sh

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

Both build scripts are **idempotent and safe to re-run**:

- `build_packages.sh` looks up each package's expected filename and size via `apt-get download --print-uris`, skips files already on disk at the right size, and replaces stale (truncated/corrupt) ones. Wheels are validated with a zip/tar integrity check before `pip download` runs, so partially-downloaded files from interrupted builds get re-fetched instead of silently shipped.
- `build_models.sh` checks each HF repo for `refs/main`, complete snapshots, no `*.incomplete` blobs, and at least one ≥ 10 MiB blob (the latter rejects "anonymous-fetch-only-README" snapshots when auth is missing). Pass `FORCE_REDOWNLOAD=1` to bypass.

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
| `WHISPER_MODELS` | `tiny base small medium large-v3 distil-large-v3` | Whisper variants to grab. Names map to `Systran/faster-whisper-*` except `distil-*` which use `Systran/faster-distil-whisper-*` (different word order). |
| `WHISPER_TURBO_REPO` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Override the large-v3-turbo source repo |
| `GEMMA_MODELS` | `google/gemma-3-1b-it google/translategemma-4b-it google/translategemma-12b-it google/translategemma-27b-it google/gemma-4-27b-it` | Gemma variants to grab; covers every HF-hosted translate preset. Set empty to skip translation models. |
| `FORCE_REDOWNLOAD` | `0` | Skip the per-repo completeness check and re-fetch every model |
| `OFFLINE_DIR` | `<repo>/offline_packages` | Override output location |

**Auth requirement.** Run `hf auth login` (or export `HF_TOKEN`) **before** calling this script. The script only sets `HF_HUB_CACHE` (the cache location) and leaves `HF_HOME` at its default, so the user-level token stored in `~/.cache/huggingface/token` is found. The script logs the token source on startup; if it warns "no HF_TOKEN env var and no token at …", gated Gemma repos will fetch only public files (44 KB README/config) and look "cached" on the next run until you authenticate and re-fetch.

Models live in HuggingFace cache layout under `offline_packages/hf-cache/hub/models--<org>--<name>/...`, so `install.sh` can copy that directly into `~/.cache/huggingface/` on the target.

---

## Phase 2: install on the offline target

Layout assumption: `~/video_ai_subtitle/` is checked out and `~/offline_packages_parts/` is its sibling (matching what `scp -r` produced in Phase 1 step 3). `tar` and `gzip` are needed (both ship with Ubuntu by default); `pv` and `pigz` are not required on the target.

```bash
cd ~/video_ai_subtitle
cat ../offline_packages_parts/bundle.tgz.* | tar xzf -    # restores ./offline_packages/
bash offline_packages/install.sh
sudo reboot                    # only if a fresh NVIDIA driver was installed
```

Add `v` to the tar flags (`tar xzvf -`) if you want filename-by-filename progress output during extraction.

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
| `hf-cache/` Whisper variants (tiny..large-v3 + turbo + distil) | ~10 GB |
| `hf-cache/` + Gemma 3 1B + TranslateGemma 4B | +12 GB |
| `hf-cache/` + TranslateGemma 12B (int8 — `balanced` preset) | +25 GB |
| `hf-cache/` + TranslateGemma 27B (int4 — `quality` preset) | +54 GB |
| `hf-cache/` + Gemma 4 27B (`gemma4-flagship` preset) | +54 GB |
| **Default total (all presets)** | **~160 GB** |

A minimal but useful bundle (transcribe-only, single Whisper model, no translation) clocks around **5 GB**. A translate-capable kit with the 4B model and Whisper turbo is around **15 GB**. The two 27B variants are the dominant cost — `GEMMA_MODELS="google/gemma-3-1b-it google/translategemma-4b-it google/translategemma-12b-it"` drops the bundle to ~50 GB while still covering `edge`, `fast`, and `balanced` translate presets.

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
