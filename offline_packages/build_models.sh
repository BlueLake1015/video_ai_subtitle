#!/usr/bin/env bash
# Download Whisper + Gemma model weights into a portable Hugging Face cache,
# so an air-gapped target can run inference without network access.
#
# Output layout (compatible with HF_HOME):
#   offline_packages/hf-cache/hub/models--<org>--<name>/...
#
# Run on an ONLINE machine that already has the project installed (.venv with
# huggingface-cli) or has python+huggingface_hub available. By default this
# uses HF_TOKEN / huggingface-cli login from the current user.
#
# Sizes are *significant*. Disable variants you don't need.
#   * Whisper presets: ~10 GB total (tiny..large-v3-turbo)
#   * Gemma 1B+4B+12B+27B: ~90 GB total in original precision
#
# Usage:
#   bash offline_packages/build_models.sh                       # everything we ship presets for
#   WHISPER_MODELS="medium large-v3-turbo" bash offline_packages/build_models.sh
#   GEMMA_MODELS="google/translategemma-4b-it" bash offline_packages/build_models.sh
#   GEMMA_MODELS=""  bash offline_packages/build_models.sh      # whisper only

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_DIR/offline_packages}"
HF_CACHE="$OFFLINE_DIR/hf-cache"

# Whisper variants matching configs/transcribe/*.yaml. faster-whisper consumes
# the Systran-converted CT2 weights; large-v3-turbo lives at a different org.
WHISPER_MODELS="${WHISPER_MODELS:-tiny base small medium large-v3 distil-large-v3}"
WHISPER_TURBO_REPO="${WHISPER_TURBO_REPO:-deepdml/faster-whisper-large-v3-turbo-ct2}"

# Gemma variants matching configs/translate/*.yaml. Comment out the giants
# (-12b/-27b) by setting GEMMA_MODELS to a smaller list.
GEMMA_MODELS="${GEMMA_MODELS:-google/gemma-3-1b-it google/translategemma-4b-it google/translategemma-12b-it}"

VENV="${VENV:-$PROJECT_DIR/.venv}"
# huggingface_hub >= 1.0 ships the `hf` CLI; older versions used the now-deprecated
# `huggingface-cli`. Prefer `hf` and fall back transparently.
if [[ -x "$VENV/bin/hf" ]]; then
    HF_CLI="$VENV/bin/hf"
elif [[ -x "$VENV/bin/huggingface-cli" ]]; then
    HF_CLI="$VENV/bin/huggingface-cli"
else
    HF_CLI=""
fi

log()  { printf '\033[1;36m[models]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn  ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err   ]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ -z "$HF_CLI" ]]; then
    die "neither $VENV/bin/hf nor $VENV/bin/huggingface-cli found. Run scripts/install_online.sh first to create .venv with huggingface_hub installed."
fi

mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"

log "HF cache: $HF_CACHE"
log "(set HF_TOKEN or run \`huggingface-cli login\` if any Gemma repo is gated)"

# ---------------- Whisper ----------------
if [[ -n "${WHISPER_MODELS// }" ]]; then
    log "downloading Whisper variants: $WHISPER_MODELS"
    for m in $WHISPER_MODELS; do
        repo="Systran/faster-whisper-$m"
        log "  -> $repo"
        "$HF_CLI" download "$repo" --quiet || warn "$repo failed (continuing)"
    done
    if [[ -n "$WHISPER_TURBO_REPO" ]]; then
        log "  -> $WHISPER_TURBO_REPO   (large-v3-turbo)"
        "$HF_CLI" download "$WHISPER_TURBO_REPO" --quiet || \
            warn "$WHISPER_TURBO_REPO failed -- try setting WHISPER_TURBO_REPO=mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    fi
fi

# ---------------- Gemma ----------------
if [[ -n "${GEMMA_MODELS// }" ]]; then
    log "downloading Gemma variants: $GEMMA_MODELS"
    log "  (gated repos require HF_TOKEN + license acceptance on huggingface.co)"
    for m in $GEMMA_MODELS; do
        log "  -> $m"
        if ! "$HF_CLI" download "$m" --quiet; then
            warn "$m failed -- did you accept its license at https://huggingface.co/$m ?"
        fi
    done
fi

# ---------------- summary ----------------
log "DONE. Cache size: $(du -sh "$HF_CACHE" 2>/dev/null | cut -f1)"
log "snapshots present:"
for d in "$HF_CACHE"/hub/models--*; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d" | sed 's|^models--||;s|--|/|g')
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    printf '         %-50s %s\n' "$name" "$size"
done

log "next:"
log "  tar czf bundle.tgz offline_packages/      # ship to the offline machine"
log "  bash offline_packages/install.sh           # on the target, after extracting bundle.tgz"
