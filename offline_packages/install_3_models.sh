#!/usr/bin/env bash
# Air-gapped install — STEP 3 of 3: model caches.
# Run AFTER install_2_python_packages.sh.
#
# Copies the bundled model caches into the locations the runtime expects:
#   hf-cache/         -> ~/.cache/huggingface   (Whisper + Gemma + opt-in ASR engines)
#   whisper-cache/    -> ~/.cache/whisper        (openai-whisper .pt, if bundled)
#   torch-hub-cache/  -> ~/.cache/torch          (whisperx align + Silero VAD, if bundled)
#
# Usage:
#   bash offline_packages/install_3_models.sh
#
# Knobs (env vars):
#   OFFLINE_DIR=path        bundle location (default $PROJECT_DIR/offline_packages)
#   VENV_DIR=path           venv location (default $PROJECT_DIR/.venv) -- for the final hint
#   HF_CACHE_TARGET=path    where to copy the HF cache (default ~/.cache/huggingface)
#   WHISPER_CACHE_TARGET=p  where to copy the openai-whisper .pt cache (default ~/.cache/whisper)
#   TORCH_HOME_TARGET=path  where to copy the whisperx torch.hub cache (default ~/.cache/torch)
#   DRY_RUN=1               preview, no changes

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_DIR/offline_packages}"

# Shared build profile (assign-if-unset; explicit env overrides win).
TARGET_ENV="${TARGET_ENV:-$PROJECT_DIR/offline_packages/target.env}"
[[ -f "$TARGET_ENV" ]] && source "$TARGET_ENV"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
HF_CACHE_TARGET="${HF_CACHE_TARGET:-$HOME/.cache/huggingface}"
# Non-HF caches for the optional openai-whisper / whisperx engines (only copied
# if the bundle contains them).
WHISPER_CACHE_TARGET="${WHISPER_CACHE_TARGET:-$HOME/.cache/whisper}"   # openai-whisper .pt
TORCH_HOME_TARGET="${TORCH_HOME_TARGET:-$HOME/.cache/torch}"           # whisperx torch.hub
DRY_RUN="${DRY_RUN:-0}"

log()  { printf '\033[1;36m[install 3/3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn       ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err        ]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then printf '\033[2m+ %s\033[0m\n' "$*"; else eval "$@"; fi; }

setup_user_run() {
    if [[ -n "${SUDO_USER:-}" && "$EUID" -eq 0 ]]; then
        USER_RUN="sudo -u $SUDO_USER -H"
        log "running under sudo as user '$SUDO_USER' -- copies will run as that user"
    else
        USER_RUN=""
    fi
}

log "project:     $PROJECT_DIR"
log "offline dir: $OFFLINE_DIR"
[[ -d "$OFFLINE_DIR" ]] || die "$OFFLINE_DIR not found. Did you transfer the bundle?"

HF_CACHE_SRC="$OFFLINE_DIR/hf-cache"
setup_user_run

# Copy a cache directory ($1) into a target ($2) if it's present in the bundle.
copy_cache() {
    local src="$1" dest="$2" label="$3"
    [[ -d "$src" ]] || { log "[$label] $src not in bundle, skipping"; return 0; }
    log "copying $label cache: $src -> $dest"
    run "mkdir -p '$dest'"
    if command -v rsync >/dev/null 2>&1; then
        run "$USER_RUN rsync -a '$src/' '$dest/'"
    else
        run "$USER_RUN cp -rn '$src/.' '$dest/'"
    fi
}

# ---------------- HuggingFace cache (Whisper + Gemma + opt-in ASR engines) ----------------
copy_cache "$HF_CACHE_SRC" "$HF_CACHE_TARGET" "HuggingFace"
[[ -d "$HF_CACHE_TARGET" ]] && log "HF cache size: $(du -sh "$HF_CACHE_TARGET" 2>/dev/null | cut -f1)"

# ---------------- optional engine caches (openai-whisper / whisperx) ----------------
copy_cache "$OFFLINE_DIR/whisper-cache"   "$WHISPER_CACHE_TARGET" "openai-whisper (.pt)"
copy_cache "$OFFLINE_DIR/torch-hub-cache" "$TORCH_HOME_TARGET"    "whisperx torch.hub"

# ---------------- summary ----------------
echo
log "DONE (step 3/3: model caches). Offline install complete."
log "next steps:"
log "  1. (if a kernel-module .deb was installed in step 1) sudo reboot   # first time only"
log "  2. source $VENV_DIR/bin/activate"
log "  3. nvidia-smi                       # verify GPU access"
log "  4. vas list-presets                 # sanity-check the install"
log "  5. vas subtitle tests/fixtures/test_en_30s.mp4 -o /tmp/en.srt -t medium --src-lang en"
