#!/usr/bin/env bash
# Profile A installer for video_ai_subtitle: NVIDIA driver only (no CUDA toolkit).
# Target: Ubuntu 22.04 / 24.04 with an Ada / Hopper / newer NVIDIA GPU (e.g. RTX 4090).
#
# What this does:
#   1. Verifies OS and GPU are present.
#   2. Installs (or upgrades) the NVIDIA driver via Ubuntu's apt repo.
#   3. Verifies nvidia-smi.
#   4. Installs Python 3.11 (deadsnakes PPA on 22.04) if missing.
#   5. Creates .venv with python3.11.
#   6. Detects the driver's max-supported CUDA via nvidia-smi and installs the
#      matching PyTorch wheel (cu128 / cu124 / cu121 / cu118 / cpu). Override
#      with TORCH_CUDA=cu124 etc.
#   7. pip-installs the project with "[translate,quant,dev]" extras.
#      Recreates an existing venv if it was built with the wrong Python.
#
# What this does NOT do:
#   * Install the CUDA toolkit (`nvcc`) -- pip wheels bundle CUDA runtime.
#   * Install cuDNN / TensorRT system-wide.
#   * Reboot for you. A reboot is required after a driver install/upgrade
#     before nvidia-smi will work; the script tells you when.
#
# Re-run safely: idempotent. Skips driver install when current >= TARGET.

set -euo pipefail

# ---------------- tunables (override via env) ----------------
TARGET_DRIVER="${TARGET_DRIVER:-560}"   # 555 / 560 / 565 all fine for RTX 4090
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PIP_EXTRAS="${PIP_EXTRAS:-translate,quant,dev}"
SKIP_PYTHON="${SKIP_PYTHON:-0}"         # 1 = don't touch Python env
DRY_RUN="${DRY_RUN:-0}"                 # 1 = print commands, don't execute
PYTHON_VERSION="${PYTHON_VERSION:-3.11}" # required by pyproject.toml (>=3.11)
RECREATE_VENV="${RECREATE_VENV:-auto}"  # auto | always | never
TORCH_CUDA="${TORCH_CUDA:-auto}"        # auto | cu130 | cu128 | cu124 | cu121 | cu118 | cpu

# ---------------- helpers ----------------
log()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '\033[2m+ %s\033[0m\n' "$*"
    else
        eval "$@"
    fi
}

need_sudo() {
    if [[ $EUID -eq 0 ]]; then
        SUDO=""
    elif command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        die "need root or sudo to install packages"
    fi
}

# Pick the right PyTorch wheel index for the installed driver. Returns one of:
#   cu128 / cu124 / cu121 / cu118 / cpu
# PyTorch built for a newer CUDA than the driver supports refuses to run with
# "RuntimeError: The NVIDIA driver on your system is too old (found version ...)".
# So we map the driver's max-supported CUDA down to the closest available wheel
# index. If TORCH_CUDA is set to anything other than "auto", we honor it.
detect_torch_cuda_index() {
    if [[ "$TORCH_CUDA" != "auto" ]]; then
        echo "$TORCH_CUDA"
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "cpu"
        return
    fi
    # nvidia-smi prints "... CUDA Version: 12.8     |" -- the trailing "|" is
    # the table border, not data. Use a regex to extract just the number.
    local cuda_str
    cuda_str=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
    if [[ ! "$cuda_str" =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "cpu"
        return
    fi
    local major minor
    major=${cuda_str%%.*}
    minor=${cuda_str#*.}
    # cu128 is the safest "modern" wheel as of April 2026 -- works on any
    # driver >= ~555. We don't pick cu130 even on driver-580+ because not
    # every consumer (e.g. ctranslate2 4.5..4.7) ships cu130 builds.
    if (( major > 12 )); then
        echo "cu128"
    elif (( major == 12 && minor >= 8 )); then
        echo "cu128"
    elif (( major == 12 && minor >= 4 )); then
        echo "cu124"
    elif (( major == 12 && minor >= 1 )); then
        echo "cu121"
    elif (( major == 11 && minor >= 8 )); then
        echo "cu118"
    else
        echo "cpu"
    fi
}

# When the whole script was invoked under sudo, USER_RUN drops privileges back
# to $SUDO_USER for venv / pip / file-ownership-sensitive operations. Otherwise
# the script runs as the user and USER_RUN is a no-op.
setup_user_run() {
    if [[ -n "${SUDO_USER:-}" && "$EUID" -eq 0 ]]; then
        USER_RUN="sudo -u $SUDO_USER -H"
        log "running under sudo as user '$SUDO_USER' -- venv/pip will run as that user"
    elif [[ "$EUID" -eq 0 ]]; then
        USER_RUN=""
        warn "running as root with no SUDO_USER set -- venv will be root-owned"
    else
        USER_RUN=""
    fi
}

# ---------------- preflight ----------------
log "Profile A installer for video_ai_subtitle"
log "project dir: $PROJECT_DIR"
log "target driver: $TARGET_DRIVER"
[[ "$DRY_RUN" == "1" ]] && warn "DRY_RUN=1 -- printing commands only"

# OS check
if [[ ! -r /etc/os-release ]]; then
    die "/etc/os-release missing -- this script targets Ubuntu"
fi
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
    die "unsupported OS: ${ID:-unknown} (this script targets Ubuntu)"
fi
case "${VERSION_ID:-}" in
    22.04|24.04) log "OS: Ubuntu ${VERSION_ID}" ;;
    *) warn "untested Ubuntu version: ${VERSION_ID} -- proceeding, but YMMV" ;;
esac

# GPU presence
if command -v lspci >/dev/null 2>&1; then
    if ! lspci -nn | grep -i -E 'nvidia|10de:' >/dev/null; then
        warn "no NVIDIA PCI device detected via lspci"
        warn "if you're sure a GPU is present, IOMMU/passthrough may be hiding it"
    else
        log "NVIDIA GPU detected:"
        lspci -nn | grep -i -E 'nvidia|10de:' | sed 's/^/         /'
    fi
else
    warn "lspci not installed -- skipping GPU presence check"
fi

# Existing driver. nvidia-smi may be present but the kernel module not loaded
# (e.g. mid-install, missing reboot); in that case it prints an error string
# rather than a version. Only accept output that *looks* like a version.
EXISTING_DRIVER=""
if command -v nvidia-smi >/dev/null 2>&1; then
    raw=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    if [[ "$raw" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
        EXISTING_DRIVER="$raw"
        log "existing nvidia driver: $EXISTING_DRIVER"
    else
        log "nvidia-smi present but driver not reporting a version (probably needs reboot)"
    fi
fi

need_sudo
setup_user_run

# ---------------- driver install ----------------
SHOULD_INSTALL=1
if [[ -n "$EXISTING_DRIVER" ]]; then
    EXISTING_MAJOR="${EXISTING_DRIVER%%.*}"
    if [[ -n "$EXISTING_MAJOR" && "$EXISTING_MAJOR" -ge "$TARGET_DRIVER" ]]; then
        log "current driver ($EXISTING_DRIVER) >= target ($TARGET_DRIVER); skipping install"
        SHOULD_INSTALL=0
    else
        log "driver $EXISTING_DRIVER < target $TARGET_DRIVER; will upgrade"
    fi
fi

if [[ "$SHOULD_INSTALL" == "1" ]]; then
    log "updating apt index"
    run "$SUDO apt-get update -y"

    log "installing prerequisites"
    run "$SUDO apt-get install -y --no-install-recommends ca-certificates curl gnupg pciutils"

    # Use Ubuntu's packaged drivers from the standard archive, not graphics-drivers PPA.
    # nvidia-driver-XXX is meta-package pulling kernel module + userland.
    PKG="nvidia-driver-${TARGET_DRIVER}"
    log "installing $PKG (this can take several minutes)"
    if ! run "$SUDO apt-get install -y $PKG"; then
        warn "$PKG not found in default archive; falling back to ubuntu-drivers"
        run "$SUDO apt-get install -y ubuntu-drivers-common"
        run "$SUDO ubuntu-drivers install"
    fi

    REBOOT_REQUIRED=1
else
    REBOOT_REQUIRED=0
fi

# ---------------- ffmpeg (project requirement) ----------------
if ! command -v ffmpeg >/dev/null 2>&1; then
    log "installing ffmpeg"
    run "$SUDO apt-get install -y ffmpeg"
else
    log "ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
fi

# ---------------- Python 3.11 + venv + project ----------------
if [[ "$SKIP_PYTHON" != "1" ]]; then
    PY="python${PYTHON_VERSION}"

    # Step 1: ensure python3.11 binary exists.
    if ! command -v "$PY" >/dev/null 2>&1; then
        log "$PY not found; installing"
        case "${VERSION_ID:-}" in
            22.04)
                # deadsnakes PPA carries python3.11 for jammy.
                run "$SUDO apt-get install -y software-properties-common"
                run "$SUDO add-apt-repository -y ppa:deadsnakes/ppa"
                run "$SUDO apt-get update -y"
                run "$SUDO apt-get install -y ${PY} ${PY}-venv ${PY}-dev"
                ;;
            24.04)
                # 24.04 ships python3.12; deadsnakes also publishes for noble.
                if ! run "$SUDO apt-get install -y ${PY} ${PY}-venv ${PY}-dev"; then
                    run "$SUDO apt-get install -y software-properties-common"
                    run "$SUDO add-apt-repository -y ppa:deadsnakes/ppa"
                    run "$SUDO apt-get update -y"
                    run "$SUDO apt-get install -y ${PY} ${PY}-venv ${PY}-dev"
                fi
                ;;
            *)
                run "$SUDO apt-get install -y ${PY} ${PY}-venv ${PY}-dev" || \
                    die "couldn't install $PY -- install it manually and re-run with PY_BIN=...'"
                ;;
        esac
    else
        log "$PY already present: $($PY --version 2>&1)"
    fi

    # Step 2: ensure venv module is available.
    run "$SUDO apt-get install -y ${PY}-venv ${PY}-dev"

    # Step 3: decide whether existing venv is reusable.
    DO_RECREATE=0
    if [[ -d "$VENV_DIR" ]]; then
        VENV_PY="$VENV_DIR/bin/python"
        if [[ -x "$VENV_PY" ]]; then
            VENV_VER=$("$VENV_PY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")
            log "existing venv python: $VENV_VER"
            if [[ "$VENV_VER" != "$PYTHON_VERSION" ]]; then
                case "$RECREATE_VENV" in
                    never)  warn "venv is $VENV_VER, want $PYTHON_VERSION, but RECREATE_VENV=never -- continuing as-is" ;;
                    *)      log "venv is $VENV_VER, want $PYTHON_VERSION -- recreating"; DO_RECREATE=1 ;;
                esac
            fi
        else
            DO_RECREATE=1
        fi
        if [[ "$RECREATE_VENV" == "always" ]]; then
            DO_RECREATE=1
        fi
    fi
    if [[ "$DO_RECREATE" == "1" ]]; then
        log "removing $VENV_DIR"
        run "rm -rf '$VENV_DIR'"
    fi

    # Step 4: create venv if needed (always as the target user, never as root).
    if [[ ! -d "$VENV_DIR" ]]; then
        log "creating venv at $VENV_DIR with $PY"
        run "$USER_RUN $PY -m venv '$VENV_DIR'"
    fi

    log "upgrading pip in venv"
    run "$USER_RUN '$VENV_DIR/bin/pip' install --upgrade pip wheel setuptools"

    # Install PyTorch from the CUDA-matched wheel index BEFORE the project, so
    # the project's transitive `torch>=2.2` doesn't pull cu130 from default PyPI
    # (which would fail at runtime on driver-570 / CUDA 12.8 boxes).
    if [[ ",$PIP_EXTRAS," == *,translate,* || ",$PIP_EXTRAS," == *,all,* ]]; then
        TCUDA=$(detect_torch_cuda_index)
        if [[ "$TCUDA" == "cpu" ]]; then
            log "installing PyTorch (CPU-only build -- no CUDA driver detected)"
            run "$USER_RUN '$VENV_DIR/bin/pip' install --index-url https://download.pytorch.org/whl/cpu torch torchaudio"
        else
            log "installing PyTorch ($TCUDA build, matched to driver)"
            run "$USER_RUN '$VENV_DIR/bin/pip' install --index-url https://download.pytorch.org/whl/$TCUDA torch torchaudio"
        fi
    fi

    log "installing project (extras: $PIP_EXTRAS)"
    run "$USER_RUN '$VENV_DIR/bin/pip' install -e '$PROJECT_DIR'[$PIP_EXTRAS]"
fi

# ---------------- verification ----------------
echo
log "verification"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        nvidia-smi | sed 's/^/         /'
    else
        warn "nvidia-smi present but reports an error -- reboot likely required"
        REBOOT_REQUIRED=1
    fi
else
    warn "nvidia-smi not on PATH yet -- reboot likely required"
    REBOOT_REQUIRED=1
fi

# ---------------- done ----------------
echo
if [[ "$REBOOT_REQUIRED" == "1" ]]; then
    log "DONE -- reboot required before GPU is usable: sudo reboot"
else
    log "DONE -- GPU is ready"
fi
log "next steps:"
log "  1. (if reboot needed) sudo reboot"
log "  2. source $VENV_DIR/bin/activate"
log "  3. vas list-presets"
log "  4. vas subtitle input.mp4 -o out.srt -t large-v3-turbo"
