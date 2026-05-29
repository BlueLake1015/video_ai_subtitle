#!/usr/bin/env bash
# Air-gapped install — STEP 1 of 3: apt/.deb packages.
# Run on the OFFLINE target, in order: install_1_ubuntu_packages.sh ->
# install_2_python_packages.sh -> install_3_models.sh.
#
# Installs the NVIDIA driver .debs (nvidia-debs/, may need a reboot) and the
# generic .debs (debs/: Python 3.11, ffmpeg, libraries, kernel headers) that
# were gathered by download_1_ubuntu_packages.sh. apt resolves dependency order
# from the local pool.
#
# Usage:
#   bash offline_packages/install_1_ubuntu_packages.sh
#
# Knobs (env vars):
#   OFFLINE_DIR=path        bundle location (default $PROJECT_DIR/offline_packages)
#   SKIP_NVIDIA_DEBS=1      don't install nvidia-debs (target already has a driver)
#   DRY_RUN=1               preview, no changes

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_DIR/offline_packages}"

# Shared build profile (assign-if-unset; explicit env overrides win).
TARGET_ENV="${TARGET_ENV:-$PROJECT_DIR/offline_packages/target.env}"
[[ -f "$TARGET_ENV" ]] && source "$TARGET_ENV"

SKIP_NVIDIA_DEBS="${SKIP_NVIDIA_DEBS:-0}"
DRY_RUN="${DRY_RUN:-0}"

log()  { printf '\033[1;36m[install 1/3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn       ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err        ]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then printf '\033[2m+ %s\033[0m\n' "$*"; else eval "$@"; fi; }

need_sudo() {
    if [[ $EUID -eq 0 ]]; then SUDO="";
    elif command -v sudo >/dev/null 2>&1; then SUDO="sudo";
    else die "need root or sudo to install .deb packages"; fi
}

log "project:     $PROJECT_DIR"
log "offline dir: $OFFLINE_DIR"
[[ -d "$OFFLINE_DIR" ]] || die "$OFFLINE_DIR not found. Did you transfer the bundle?"

DEBS_DIR="$OFFLINE_DIR/debs"
NVIDIA_DEBS_DIR="$OFFLINE_DIR/nvidia-debs"
need_sudo

# Install all .debs in a directory (no-op if dir is missing or empty).
install_debs_dir() {
    local dir="$1"; local label="$2"
    if [[ ! -d "$dir" ]]; then
        log "[$label] $dir missing, skipping"
        return 0
    fi
    local n
    n=$(find "$dir" -maxdepth 1 -name '*.deb' 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then
        log "[$label] no .deb files in $dir, skipping"
        return 0
    fi
    log "[$label] installing $n .deb packages from $dir"
    # apt-get install resolves dependency order from the local pool when given
    # explicit file paths.
    run "$SUDO apt-get install -y --no-install-recommends '$dir'/*.deb"
}

# ---------------- 1a. NVIDIA driver .debs (install first; may need reboot) ----------------
if [[ "$SKIP_NVIDIA_DEBS" != "1" ]]; then
    install_debs_dir "$NVIDIA_DEBS_DIR" "nvidia"
else
    log "skipping nvidia-debs (SKIP_NVIDIA_DEBS=1)"
fi

# ---------------- 1b. generic .debs (python, ffmpeg, etc.) ----------------
install_debs_dir "$DEBS_DIR" "debs"

log "DONE (step 1/3: apt packages)."
log "next: bash offline_packages/install_2_python_packages.sh"
log "      (if a kernel-module .deb was installed, you may need to reboot before the GPU is usable)"
