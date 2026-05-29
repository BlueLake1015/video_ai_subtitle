#!/usr/bin/env bash
# Gather every Python wheel/sdist needed to install the project on an air-gapped
# target. Run on a machine that DOES have internet, with the same Python series
# and CUDA class as the offline target.
#
# This is the PIP half of the bundle. For apt/.deb packages run
# `download_1_ubuntu_packages.sh`. Produces:
#
#   offline_packages/
#     pip-wheels/            *.whl, *.tar.gz, requirements.txt
#     manifest_python.txt    inventory of the wheel artifacts
#
# Usage:
#   bash offline_packages/download_2_python_packages.sh
#
# Knobs (env vars):
#   PYTHON_VERSION=3.11      python series to resolve against (default 3.11)
#   TORCH_CUDA=cu128         which PyTorch wheel index (default cu128 = CUDA 12.8)
#   TORCH_VERSION=2.6.0      pin torch+torchaudio (default: latest on the index).
#                            Use cu124 + a <=2.6 pin for Volta/sm_70 targets (V100).
#   PIP_EXTRAS=translate,quant,dev,whispercpp,gemini   extras to resolve
#   OFFLINE_DIR=path/        output directory (default $PROJECT_DIR/offline_packages)
#   DRY_RUN=1                preview, don't execute

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_DIR/offline_packages}"

# Shared build profile (target GPU / CUDA / torch / python). Values there are
# assign-if-unset, so explicit `VAR=... bash <script>` overrides still win.
TARGET_ENV="${TARGET_ENV:-$PROJECT_DIR/offline_packages/target.env}"
[[ -f "$TARGET_ENV" ]] && source "$TARGET_ENV"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
# Pin a specific torch (and matching torchaudio) version. Empty (default) takes the
# newest wheel on the $TORCH_CUDA index — correct for the RTX 4090 / sm_89 target.
# Set this to make builds reproducible, or to support an older GPU whose latest
# wheel dropped support (e.g. a Volta/sm_70 V100 needs cu124 + TORCH_VERSION=2.6.0).
TORCH_VERSION="${TORCH_VERSION:-}"
PIP_EXTRAS="${PIP_EXTRAS:-translate,quant,dev}"
DRY_RUN="${DRY_RUN:-0}"

# torchaudio releases version-lock to torch (2.6.0<->2.6.0, …), so pin both.
if [[ -n "$TORCH_VERSION" ]]; then
    TORCH_SPEC="torch==$TORCH_VERSION torchaudio==$TORCH_VERSION"
else
    TORCH_SPEC="torch torchaudio"
fi

log()  { printf '\033[1;36m[python]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn  ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err   ]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then printf '\033[2m+ %s\033[0m\n' "$*"; else eval "$@"; fi; }

mkdir -p "$OFFLINE_DIR"
WHEELS_DIR="$OFFLINE_DIR/pip-wheels"
MANIFEST="$OFFLINE_DIR/manifest_python.txt"
mkdir -p "$WHEELS_DIR"

log "project:        $PROJECT_DIR"
log "offline dir:    $OFFLINE_DIR"
log "python series:  $PYTHON_VERSION"
log "torch CUDA:     $TORCH_CUDA"
log "torch version:  ${TORCH_VERSION:-latest}"
log "pip extras:     $PIP_EXTRAS"

# ---------------- pip wheels ----------------
PY="python${PYTHON_VERSION}"
command -v "$PY" >/dev/null || die "$PY not on PATH (needed to resolve wheels)"

TMPVENV="$OFFLINE_DIR/.tmp-resolve-venv"
log "creating temporary venv at $TMPVENV (used only to resolve dep versions)"
run "rm -rf '$TMPVENV'"
run "$PY -m venv '$TMPVENV'"
run "'$TMPVENV/bin/pip' install --upgrade -q pip wheel setuptools"

# `pip freeze` (step 3) omits pip/wheel/setuptools by design, so they would
# never make it into the bundle on their own. install_2_python_packages.sh upgrades
# them from local wheels before installing the rest, so seed them here.
log "downloading pip/wheel/setuptools into $WHEELS_DIR for offline bootstrap"
run "'$TMPVENV/bin/pip' download -q -d '$WHEELS_DIR' pip wheel setuptools"

# Step 1: install torch+torchaudio from CUDA-specific index so freeze records
# the right +cuXXX local version specifier.
log "installing torch ($TORCH_CUDA, ${TORCH_VERSION:-latest}) into resolver venv"
run "'$TMPVENV/bin/pip' install -q --index-url https://download.pytorch.org/whl/$TORCH_CUDA $TORCH_SPEC"

# Step 2: install project + extras. Must keep the CUDA wheel index reachable
# via --extra-index-url, otherwise pip re-resolves the [translate] extra's
# `torch>=2.2` against default PyPI and silently swaps the +cuXXX wheel from
# step 1 for the same-canonical-version +cpu wheel. That would then get
# baked into requirements.txt by `pip freeze` in step 3 and shipped to the
# air-gapped target as a broken (CPU-only) bundle.
log "installing project + extras into resolver venv"
run "'$TMPVENV/bin/pip' install -q --extra-index-url https://download.pytorch.org/whl/$TORCH_CUDA -e '$PROJECT_DIR'[$PIP_EXTRAS]"

# Step 3: produce a frozen requirements list (excluding the project itself).
REQ="$WHEELS_DIR/requirements.txt"
log "freezing resolved versions to $REQ"
if [[ "$DRY_RUN" != "1" ]]; then
    "$TMPVENV/bin/pip" freeze \
        | grep -v '^-e \|^video-ai-subtitle\|^pkg-resources==' \
        > "$REQ"
fi

# Step 3.5: verify wheels already in WHEELS_DIR (from a previous run).
# `pip download -d DIR` skips files with matching name+version but does NOT
# validate their contents -- a previous Ctrl-C, network drop, or full-disk
# event can leave a truncated/empty wheel that pip will then happily skip.
# Validate zip/tar integrity now and remove anything broken so pip re-fetches.
if [[ "$DRY_RUN" != "1" ]]; then
    verified=0; corrupt=0; tiny=0
    while IFS= read -r -d '' f; do
        verified=$((verified+1))
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [[ "$sz" -lt 200 ]]; then
            warn "[wheels] tiny: $(basename "$f") ($sz B) -- removing"
            rm -f "$f"; tiny=$((tiny+1))
            continue
        fi
        ok=1
        case "$f" in
            *.whl)
                "$TMPVENV/bin/python" -c '
import sys, zipfile
try:
    z = zipfile.ZipFile(sys.argv[1])
    sys.exit(0 if z.testzip() is None else 1)
except Exception:
    sys.exit(1)
' "$f" >/dev/null 2>&1 || ok=0
                ;;
            *.tar.gz|*.tgz)
                tar -tzf "$f" >/dev/null 2>&1 || ok=0
                ;;
        esac
        if [[ "$ok" == "0" ]]; then
            warn "[wheels] corrupt: $(basename "$f") -- removing"
            rm -f "$f"; corrupt=$((corrupt+1))
        fi
    done < <(find "$WHEELS_DIR" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.tgz' \) -print0)
    log "[wheels] verified $verified existing files: removed $corrupt corrupt + $tiny tiny"
fi

# Step 4: download every wheel matching those exact versions.
# pip download skips files already on disk that match the resolved
# name+version, so the verification pass above is what guarantees freshness.
log "downloading wheels into $WHEELS_DIR"
run "'$TMPVENV/bin/pip' download \
    --index-url https://download.pytorch.org/whl/$TORCH_CUDA \
    --extra-index-url https://pypi.org/simple \
    -d '$WHEELS_DIR' \
    -r '$REQ'"

log "cleaning up resolver venv"
run "rm -rf '$TMPVENV'"

WHL_COUNT=$(find "$WHEELS_DIR" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)
SDIST_COUNT=$(find "$WHEELS_DIR" -maxdepth 1 -name '*.tar.gz' 2>/dev/null | wc -l)
log "[wheels] gathered $WHL_COUNT wheels + $SDIST_COUNT sdists ($(du -sh "$WHEELS_DIR" 2>/dev/null | cut -f1))"

# ---------------- manifest (pip artifacts) ----------------
log "writing manifest -> $MANIFEST"
{
    echo "# video_ai_subtitle offline-packages bundle (pip half)"
    echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host:      $(uname -a)"
    echo "# python:        $PYTHON_VERSION"
    echo "# torch_cuda:    $TORCH_CUDA"
    echo "# torch_version: ${TORCH_VERSION:-latest}"
    echo "# pip_extras:    $PIP_EXTRAS"
    echo
    echo "## pip-wheels/"
    (cd "$WHEELS_DIR" && ls -1 2>/dev/null) || echo "(none)"
} > "$MANIFEST"

log "DONE (pip half)."
log "next:"
log "  bash offline_packages/download_3_models.sh             # download model weights"
log ""
log "  # bundle into 1 GiB parts (sudo apt install pv pigz -- pigz is multi-threaded gzip,"
log "  # 5-10x faster than the single-threaded default; output stays gzip-compatible):"
log "  mkdir -p ../offline_packages_parts && \\"
log "    tar cf - offline_packages/ \\"
log "      | pv -s \$(du -sb offline_packages | cut -f1) \\"
log "      | pigz \\"
log "      | split -b 1G -d -a 3 - ../offline_packages_parts/bundle.tgz."
