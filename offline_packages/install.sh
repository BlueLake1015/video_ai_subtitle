#!/usr/bin/env bash
# Air-gapped installer: install everything from a previously-built
# offline_packages/ bundle. Run on the OFFLINE target machine.
#
# Prerequisites:
#   * The offline_packages/ directory has been populated on an online machine
#     via offline_packages/build_packages.sh and (optionally) build_offline_models.sh,
#     then transferred to this host (e.g. as bundle.tgz).
#   * Target is Ubuntu 22.04 / 24.04 with the same architecture (amd64).
#
# What this does:
#   1. Verifies bundle layout.
#   2. Installs every .deb from offline_packages/debs/ in dependency order
#      (apt resolves order from the local pool).
#   3. Creates a Python venv with the system python3.11.
#   4. Installs every .whl/.tar.gz from offline_packages/pip-wheels/ with
#      pip --no-index, locked to the frozen requirements.txt.
#   5. Copies offline_packages/hf-cache/ into ~/.cache/huggingface/ so models
#      are usable without network access.
#   6. Sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE in .venv/bin/activate so the
#      venv defaults to fully offline operation.
#
# Usage:
#   bash offline_packages/install.sh
#
# Knobs (env vars):
#   OFFLINE_DIR=path        bundle location (default $PROJECT_DIR/offline_packages)
#   VENV_DIR=path           venv location (default $PROJECT_DIR/.venv)
#   PYTHON_VERSION=3.11     python series to use (default 3.11)
#   HF_CACHE_TARGET=path    where to copy models (default ~/.cache/huggingface)
#   SKIP_DEBS=1             don't install apt packages (already done)
#   SKIP_PIP=1              don't install Python wheels
#   SKIP_MODELS=1           don't copy the HF cache
#   DRY_RUN=1               preview, no changes

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_DIR/offline_packages}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
HF_CACHE_TARGET="${HF_CACHE_TARGET:-$HOME/.cache/huggingface}"
SKIP_DEBS="${SKIP_DEBS:-0}"
SKIP_NVIDIA_DEBS="${SKIP_NVIDIA_DEBS:-0}"
SKIP_PIP="${SKIP_PIP:-0}"
SKIP_MODELS="${SKIP_MODELS:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"   # 1 = don't verify sha256sums.txt before installing
DRY_RUN="${DRY_RUN:-0}"

log()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn   ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err    ]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then printf '\033[2m+ %s\033[0m\n' "$*"; else eval "$@"; fi; }

need_sudo() {
    if [[ $EUID -eq 0 ]]; then SUDO="";
    elif command -v sudo >/dev/null 2>&1; then SUDO="sudo";
    else die "need root or sudo to install .deb packages"; fi
}

setup_user_run() {
    if [[ -n "${SUDO_USER:-}" && "$EUID" -eq 0 ]]; then
        USER_RUN="sudo -u $SUDO_USER -H"
        log "running under sudo as user '$SUDO_USER' -- venv/pip will run as that user"
    else
        USER_RUN=""
    fi
}

# ---------------- preflight ----------------
log "project:     $PROJECT_DIR"
log "offline dir: $OFFLINE_DIR"
log "venv:        $VENV_DIR"

[[ -d "$OFFLINE_DIR" ]] || die "$OFFLINE_DIR not found. Did you transfer the bundle?"

DEBS_DIR="$OFFLINE_DIR/debs"
NVIDIA_DEBS_DIR="$OFFLINE_DIR/nvidia-debs"
WHEELS_DIR="$OFFLINE_DIR/pip-wheels"
HF_CACHE_SRC="$OFFLINE_DIR/hf-cache"
REQ_FILE="$WHEELS_DIR/requirements.txt"
SUMS_FILE="$OFFLINE_DIR/sha256sums.txt"

need_sudo
setup_user_run

# ---------------- 0. integrity check ----------------
if [[ "$SKIP_VERIFY" != "1" ]]; then
    if [[ -f "$SUMS_FILE" ]]; then
        n=$(wc -l < "$SUMS_FILE")
        log "verifying sha256sums of $n bundle files"
        # `sha256sum -c` reads paths relative to the cwd, so cd into the bundle.
        if ! ( cd "$OFFLINE_DIR" && sha256sum -c --quiet --status "$SUMS_FILE" ); then
            warn "checksum mismatch detected; re-running with full output:"
            ( cd "$OFFLINE_DIR" && sha256sum -c "$SUMS_FILE" 2>&1 | grep -v ': OK$' ) >&2 || true
            die "bundle integrity check FAILED -- transfer corruption or tampering. Re-copy the bundle, or re-run with SKIP_VERIFY=1 to bypass."
        fi
        log "checksums OK"
    else
        warn "no sha256sums.txt in $OFFLINE_DIR -- skipping integrity check"
        warn "(modern bundles ship one; older bundles or partially-built ones may not)"
    fi
else
    log "skipping checksum verification (SKIP_VERIFY=1)"
fi

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
if [[ "$SKIP_DEBS" != "1" ]]; then
    install_debs_dir "$DEBS_DIR" "debs"
else
    log "skipping debs (SKIP_DEBS=1)"
fi

# ---------------- 2. Python venv ----------------
if [[ "$SKIP_PIP" != "1" ]]; then
    PY="python${PYTHON_VERSION}"
    command -v "$PY" >/dev/null || die "$PY not on PATH. Did the python${PYTHON_VERSION} .deb install?"
    [[ -d "$WHEELS_DIR" ]] || die "$WHEELS_DIR missing"
    [[ -f "$REQ_FILE" ]] || die "$REQ_FILE missing -- did you run build_offline_packages.sh?"

    if [[ -d "$VENV_DIR" ]]; then
        VENV_VER=$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")
        if [[ "$VENV_VER" != "$PYTHON_VERSION" ]]; then
            log "existing venv is python$VENV_VER, recreating with $PY"
            run "rm -rf '$VENV_DIR'"
        fi
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        log "creating venv: $PY -m venv $VENV_DIR"
        run "$USER_RUN $PY -m venv '$VENV_DIR'"
    fi

    # bootstrap pip from a wheel that's in the bundle (pip wheel is normally
    # downloaded by ensurepip; we override with the bundled one to be safe).
    log "upgrading pip from local wheels"
    run "$USER_RUN '$VENV_DIR/bin/pip' install --no-index --find-links '$WHEELS_DIR' --upgrade pip wheel setuptools"

    log "installing project dependencies from $WHEELS_DIR ($(wc -l < "$REQ_FILE") packages)"
    run "$USER_RUN '$VENV_DIR/bin/pip' install --no-index --find-links '$WHEELS_DIR' -r '$REQ_FILE'"

    log "installing project itself (editable, no extras since deps already present)"
    run "$USER_RUN '$VENV_DIR/bin/pip' install --no-index --no-deps --no-build-isolation -e '$PROJECT_DIR'"
else
    log "skipping pip (SKIP_PIP=1)"
fi

# ---------------- 3. Models ----------------
if [[ "$SKIP_MODELS" != "1" ]] && [[ -d "$HF_CACHE_SRC" ]]; then
    log "copying HF cache: $HF_CACHE_SRC -> $HF_CACHE_TARGET"
    run "mkdir -p '$HF_CACHE_TARGET'"
    if command -v rsync >/dev/null 2>&1; then
        run "$USER_RUN rsync -a --info=progress2 '$HF_CACHE_SRC/' '$HF_CACHE_TARGET/'"
    else
        run "$USER_RUN cp -rn '$HF_CACHE_SRC/.' '$HF_CACHE_TARGET/'"
    fi
    log "HF cache populated: $(du -sh "$HF_CACHE_TARGET" 2>/dev/null | cut -f1)"
else
    log "skipping models (SKIP_MODELS=1 or $HF_CACHE_SRC missing)"
fi

# ---------------- 4. activate-script offline defaults ----------------
ACTIVATE="$VENV_DIR/bin/activate"
if [[ -f "$ACTIVATE" ]] && ! grep -q 'HF_HUB_OFFLINE' "$ACTIVATE"; then
    log "appending offline env vars to $ACTIVATE"
    if [[ "$DRY_RUN" != "1" ]]; then
        cat >> "$ACTIVATE" <<'EOF'

# Added by offline_packages/install.sh -- prevents accidental network lookups.
# Override with `HF_HUB_OFFLINE=0 vas ...` if you want to enable HF downloads.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF
    fi
fi

# ---------------- summary ----------------
echo
log "DONE."
log "next steps:"
log "  1. (if a kernel-module .deb was installed) sudo reboot   # only needed first time"
log "  2. source $VENV_DIR/bin/activate"
log "  3. nvidia-smi                       # verify GPU access"
log "  4. vas list-presets                 # sanity-check the install"
log "  5. vas subtitle tests/fixtures/test_en_30s.mp4 -o /tmp/en.srt -t medium --src-lang en"
