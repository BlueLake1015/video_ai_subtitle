#!/usr/bin/env bash
# Profile A installer for video_ai_subtitle: NVIDIA driver only (no CUDA toolkit).
# Target: Ubuntu 22.04 / 24.04 with an Ada / Hopper / newer NVIDIA GPU (e.g. RTX 4090).
#
# What this does:
#   1. Verifies OS and GPU are present.
#   2. Installs (or upgrades) the NVIDIA driver via Ubuntu's apt repo.
#   3. Verifies nvidia-smi.
#   4. Optionally creates .venv and pip-installs the project with
#      "[translate,quant,dev]" extras.
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

# Existing driver
EXISTING_DRIVER=""
if command -v nvidia-smi >/dev/null 2>&1; then
    EXISTING_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    log "existing nvidia driver: ${EXISTING_DRIVER:-<none reported>}"
fi

need_sudo

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

# ---------------- python venv + project ----------------
if [[ "$SKIP_PYTHON" != "1" ]]; then
    log "ensuring python3-venv and python3-pip"
    run "$SUDO apt-get install -y python3-venv python3-pip"

    if [[ ! -d "$VENV_DIR" ]]; then
        log "creating venv at $VENV_DIR"
        run "python3 -m venv '$VENV_DIR'"
    fi

    log "upgrading pip in venv"
    run "'$VENV_DIR/bin/pip' install --upgrade pip wheel setuptools"

    log "installing project (extras: $PIP_EXTRAS)"
    run "'$VENV_DIR/bin/pip' install -e '$PROJECT_DIR'[$PIP_EXTRAS]"
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
