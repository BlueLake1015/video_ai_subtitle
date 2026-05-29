#!/usr/bin/env bash
# Air-gapped install — STEP 2 of 3: Python venv + wheels.
# Run AFTER install_1_ubuntu_packages.sh (needs the system python3.11 it installs).
#
# Creates the .venv, installs every .whl/.tar.gz from offline_packages/pip-wheels/
# with pip --no-index (locked to the frozen requirements.txt), installs the project
# itself, and sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE in the venv's activate
# script so it defaults to fully offline operation.
#
# Usage:
#   bash offline_packages/install_2_python_packages.sh
#
# Knobs (env vars):
#   OFFLINE_DIR=path        bundle location (default $PROJECT_DIR/offline_packages)
#   VENV_DIR=path           venv location (default $PROJECT_DIR/.venv)
#   PYTHON_VERSION=3.11     python series to use (default from target.env / 3.11)
#   DRY_RUN=1               preview, no changes

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_DIR/offline_packages}"

# Shared build profile — PYTHON_VERSION here keeps the install venv consistent
# with what was bundled. Assign-if-unset; explicit env overrides win.
TARGET_ENV="${TARGET_ENV:-$PROJECT_DIR/offline_packages/target.env}"
[[ -f "$TARGET_ENV" ]] && source "$TARGET_ENV"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
DRY_RUN="${DRY_RUN:-0}"

log()  { printf '\033[1;36m[install 2/3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn       ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err        ]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then printf '\033[2m+ %s\033[0m\n' "$*"; else eval "$@"; fi; }

setup_user_run() {
    if [[ -n "${SUDO_USER:-}" && "$EUID" -eq 0 ]]; then
        USER_RUN="sudo -u $SUDO_USER -H"
        log "running under sudo as user '$SUDO_USER' -- venv/pip will run as that user"
    else
        USER_RUN=""
    fi
}

log "project:     $PROJECT_DIR"
log "offline dir: $OFFLINE_DIR"
log "venv:        $VENV_DIR"
[[ -d "$OFFLINE_DIR" ]] || die "$OFFLINE_DIR not found. Did you transfer the bundle?"

WHEELS_DIR="$OFFLINE_DIR/pip-wheels"
REQ_FILE="$WHEELS_DIR/requirements.txt"
setup_user_run

PY="python${PYTHON_VERSION}"
command -v "$PY" >/dev/null || die "$PY not on PATH. Run install_1_ubuntu_packages.sh first (it installs python${PYTHON_VERSION})."
[[ -d "$WHEELS_DIR" ]] || die "$WHEELS_DIR missing"
[[ -f "$REQ_FILE" ]] || die "$REQ_FILE missing -- did you run download_2_python_packages.sh?"

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

# bootstrap pip/setuptools/wheel from local wheels if they were bundled.
# pip freeze (used by download_2_python_packages.sh) excludes these by design, so older
# bundles may not contain them. Skip silently when missing -- they're not
# required for installing pre-built .whl files (wheel is only needed for
# building sdists, which we don't ship).
log "upgrading pip/setuptools/wheel from local wheels (where available)"
for pkg in pip setuptools wheel; do
    # check the bundle has at least one matching wheel before invoking pip,
    # so pip's scary "No matching distribution" ERROR never appears.
    if compgen -G "$WHEELS_DIR/${pkg}-*.whl" >/dev/null; then
        run "$USER_RUN '$VENV_DIR/bin/pip' install --no-index --find-links '$WHEELS_DIR' --upgrade $pkg"
    else
        warn "[bootstrap] $pkg not in bundle, leaving venv default"
    fi
done

log "installing project dependencies from $WHEELS_DIR ($(wc -l < "$REQ_FILE") packages)"
run "$USER_RUN '$VENV_DIR/bin/pip' install --no-index --find-links '$WHEELS_DIR' -r '$REQ_FILE'"

log "installing project itself (editable, no extras since deps already present)"
run "$USER_RUN '$VENV_DIR/bin/pip' install --no-index --no-deps --no-build-isolation -e '$PROJECT_DIR'"

# ---------------- activate-script offline defaults ----------------
ACTIVATE="$VENV_DIR/bin/activate"
if [[ -f "$ACTIVATE" ]] && ! grep -q 'HF_HUB_OFFLINE' "$ACTIVATE"; then
    log "appending offline env vars to $ACTIVATE"
    if [[ "$DRY_RUN" != "1" ]]; then
        cat >> "$ACTIVATE" <<'EOF'

# Added by offline_packages/install_2_python_packages.sh -- prevents accidental
# network lookups. Override with `HF_HUB_OFFLINE=0 vas ...` to allow HF downloads.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF
    fi
fi

log "DONE (step 2/3: python venv + wheels)."
log "next: bash offline_packages/install_3_models.sh"
