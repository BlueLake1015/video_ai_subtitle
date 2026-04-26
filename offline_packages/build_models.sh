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

# Gemma variants matching configs/translate/*.yaml. Includes every HF-hosted
# model referenced by a translate preset:
#   edge            -> google/gemma-3-1b-it
#   fast            -> google/translategemma-4b-it
#   balanced        -> google/translategemma-12b-it
#   quality         -> google/translategemma-27b-it
#   gemma4-flagship -> google/gemma-4-27b-it
# (ollama-12b and cloud-gemini don't use the HF cache.)
# The two 27B models are ~50 GB each and gated. Override GEMMA_MODELS to skip
# them, or set FORCE_REDOWNLOAD=1 to re-fetch a partial snapshot.
GEMMA_MODELS="${GEMMA_MODELS:-google/gemma-3-1b-it google/translategemma-4b-it google/translategemma-12b-it google/translategemma-27b-it google/gemma-4-27b-it}"

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

# Force a re-download even if the cache looks complete. Use after upstream
# revision bumps or when you've changed PROJECT_DIR/.venv's hf version.
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-0}"

if [[ -z "$HF_CLI" ]]; then
    die "neither $VENV/bin/hf nor $VENV/bin/huggingface-cli found. Run scripts/install_online.sh first to create .venv with huggingface_hub installed."
fi

mkdir -p "$HF_CACHE/hub"
# Override only the cache location, NOT HF_HOME. Setting HF_HOME=$HF_CACHE
# moves the *token file* there too -- which is empty -- so `hf download` runs
# unauthenticated even after `hf auth login`. Gated repos then silently return
# only their public files (README, config) and the snapshot looks "complete"
# at a few KB. Keep HF_HOME at its default so the user's login token is found.
export HF_HUB_CACHE="$HF_CACHE/hub"

log "HF cache: $HF_CACHE  (HF_HUB_CACHE=$HF_HUB_CACHE)"
[[ "$FORCE_REDOWNLOAD" == "1" ]] && log "FORCE_REDOWNLOAD=1 -- re-fetching every model"

# Sanity-check auth so gated Gemma repos actually download weights, not just READMEs.
HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [[ -n "${HF_TOKEN:-}" ]]; then
    log "auth: HF_TOKEN env var is set"
elif [[ -s "$HF_TOKEN_FILE" ]]; then
    log "auth: token found at $HF_TOKEN_FILE"
else
    warn "auth: no HF_TOKEN env var and no token at $HF_TOKEN_FILE"
    warn "      gated repos (gemma, translategemma) will silently fetch only public files"
    warn "      run \`hf auth login\` first, or export HF_TOKEN=<your-token>"
fi

# Translate "org/name" into the HF cache directory layout: "models--org--name".
hf_repo_dir() {
    local repo="$1"
    echo "$HF_CACHE/hub/models--${repo//\//--}"
}

# Decide whether a repo is fully on disk and safe to skip. True only when:
#   - refs/main names a commit
#   - snapshots/<commit>/ exists and is non-empty
#   - no .incomplete files in blobs/ (left over from interrupted downloads)
#   - no zero-byte blobs (truncated transfers that the hf CLI didn't finish)
#   - at least one blob >= 10 MiB exists (rejects gated repos that "succeeded"
#     anonymously and only got README/config -- happens when auth is missing)
hf_repo_complete() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    [[ -s "$dir/refs/main" ]] || return 1
    local rev
    rev=$(<"$dir/refs/main")
    [[ -d "$dir/snapshots/$rev" ]] || return 1
    [[ -n "$(ls -A "$dir/snapshots/$rev" 2>/dev/null)" ]] || return 1
    if find "$dir/blobs" -maxdepth 1 -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
        return 1
    fi
    if find "$dir/blobs" -maxdepth 1 -type f -size 0 -print -quit 2>/dev/null | grep -q .; then
        return 1
    fi
    # Real model snapshots always have a multi-MB weight file (model.safetensors,
    # model.bin, *.gguf, ct2 model.bin). If nothing in blobs/ is bigger than 10 MiB,
    # we got README + config only -- treat as incomplete so we re-download.
    if ! find "$dir/blobs" -maxdepth 1 -type f -size +10M -print -quit 2>/dev/null | grep -q .; then
        return 1
    fi
    return 0
}

# Wipe partial-download markers from a prior run so the hf CLI starts clean.
hf_repo_cleanup_partials() {
    local dir="$1"
    [[ -d "$dir/blobs" ]] || return 0
    local n
    n=$(find "$dir/blobs" -maxdepth 1 \( -name '*.incomplete' -o -size 0 \) -print -delete 2>/dev/null | wc -l)
    [[ "$n" -gt 0 ]] && warn "removed $n partial/empty blob(s) from $(basename "$dir")"
}

# Idempotent download wrapper: skip when complete, otherwise clean partials and
# fetch. Returns non-zero if hf failed (caller already logs warn-and-continue).
hf_fetch() {
    local repo="$1"
    local dir
    dir=$(hf_repo_dir "$repo")
    if [[ "$FORCE_REDOWNLOAD" != "1" ]] && hf_repo_complete "$dir"; then
        local sz
        sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
        log "  -> $repo   [cached, $sz]"
        return 0
    fi
    hf_repo_cleanup_partials "$dir"
    log "  -> $repo"
    "$HF_CLI" download "$repo" --quiet
}

# Map a short Whisper variant name (matching configs/transcribe/*.yaml) to its
# Systran HF repo. Distil variants live under a different naming convention --
# `Systran/faster-distil-whisper-large-v3`, NOT `Systran/faster-whisper-distil-large-v3`.
whisper_repo_for() {
    case "$1" in
        distil-*) echo "Systran/faster-distil-whisper-${1#distil-}" ;;
        *)        echo "Systran/faster-whisper-$1" ;;
    esac
}

# ---------------- Whisper ----------------
if [[ -n "${WHISPER_MODELS// }" ]]; then
    log "downloading Whisper variants: $WHISPER_MODELS"
    for m in $WHISPER_MODELS; do
        repo=$(whisper_repo_for "$m")
        hf_fetch "$repo" || warn "$repo failed (continuing)"
    done
    if [[ -n "$WHISPER_TURBO_REPO" ]]; then
        hf_fetch "$WHISPER_TURBO_REPO" || \
            warn "$WHISPER_TURBO_REPO failed -- try setting WHISPER_TURBO_REPO=mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    fi
fi

# ---------------- Gemma ----------------
if [[ -n "${GEMMA_MODELS// }" ]]; then
    log "downloading Gemma variants: $GEMMA_MODELS"
    log "  (gated repos require HF_TOKEN + license acceptance on huggingface.co)"
    for m in $GEMMA_MODELS; do
        if ! hf_fetch "$m"; then
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
log "  # bundle into 1 GiB parts (sudo apt install pv pigz -- pigz is multi-threaded gzip,"
log "  # 5-10x faster than the single-threaded default; output stays gzip-compatible):"
log "  mkdir -p ../offline_packages_parts && \\"
log "    tar cf - offline_packages/ \\"
log "      | pv -s \$(du -sb offline_packages | cut -f1) \\"
log "      | pigz \\"
log "      | split -b 1G -d -a 3 - ../offline_packages_parts/bundle.tgz."
log ""
log "  # on the offline machine, after copying offline_packages_parts/ over"
log "  # (add 'v' to xzf for filename-by-filename progress: tar xzvf -):"
log "  cat offline_packages_parts/bundle.tgz.* | tar xzf -"
log "  bash offline_packages/install.sh"
