#!/usr/bin/env bash
# Gather every .deb (apt) and .whl/.tar.gz (pip) needed to install the project
# on an air-gapped target machine. Run this on a machine that DOES have
# internet access, ideally with the same Ubuntu version + GPU class as the
# offline target. Produces:
#
#   offline_packages/
#     debs/                           *.deb        (apt packages, transitive)
#     pip-wheels/                     *.whl, *.tar.gz, requirements.txt
#     manifest.txt                    inventory & versions
#
# Usage:
#   bash offline_packages/build_packages.sh
#
# Knobs (env vars):
#   TARGET_DRIVER=560        nvidia-driver major to gather (default 560)
#   PYTHON_VERSION=3.11      python series to gather   (default 3.11)
#   TORCH_CUDA=cu128         which PyTorch wheel index (default cu128 = CUDA 12.8)
#   PIP_EXTRAS=translate,quant,dev,whispercpp,gemini    extras to resolve
#   OFFLINE_DIR=path/        output directory (default $PROJECT_DIR/offline_packages)
#   SKIP_DEBS=1              don't gather .debs (already have them)
#   SKIP_PIP=1               don't gather wheels
#   DRY_RUN=1                preview, don't execute

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_DIR/offline_packages}"
# NVIDIA driver: defaults to whatever is currently loaded on this build host
# (read via nvidia-smi). Override with TARGET_DRIVER=570 to pin a specific major.
TARGET_DRIVER="${TARGET_DRIVER:-auto}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
PIP_EXTRAS="${PIP_EXTRAS:-translate,quant,dev}"
SKIP_DEBS="${SKIP_DEBS:-0}"
SKIP_NVIDIA="${SKIP_NVIDIA:-0}"   # 1 = don't gather nvidia-driver debs
SKIP_PIP="${SKIP_PIP:-0}"
DRY_RUN="${DRY_RUN:-0}"
# Target kernel for DKMS (nvidia-dkms-XXX-open builds the kernel module against
# linux-headers-$(uname -r) at install time on the offline host).
#   auto     -> use this build host's kernel (only useful if target == build host)
#   <empty>  -> don't gather kernel headers (target must already have them)
#   5.15.0-176-generic -> explicit kernel version of the offline target
TARGET_KERNEL="${TARGET_KERNEL:-auto}"

log()  { printf '\033[1;36m[bundle]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn  ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err   ]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then printf '\033[2m+ %s\033[0m\n' "$*"; else eval "$@"; fi; }

[[ -r /etc/os-release ]] || die "/etc/os-release missing -- this script targets Ubuntu"
. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "unsupported OS: ${ID:-unknown}"

mkdir -p "$OFFLINE_DIR"
DEBS_DIR="$OFFLINE_DIR/debs"
NVIDIA_DEBS_DIR="$OFFLINE_DIR/nvidia-debs"
WHEELS_DIR="$OFFLINE_DIR/pip-wheels"
MANIFEST="$OFFLINE_DIR/manifest.txt"
mkdir -p "$DEBS_DIR" "$NVIDIA_DEBS_DIR" "$WHEELS_DIR"

log "project:        $PROJECT_DIR"
log "offline dir:    $OFFLINE_DIR"
log "target driver:  $TARGET_DRIVER"
log "python series:  $PYTHON_VERSION"
log "torch CUDA:     $TORCH_CUDA"
log "pip extras:     $PIP_EXTRAS"

# Patterns for "truly NVIDIA-driver" packages -- the ones that need a reboot
# (kernel modules) plus the matched-version userland that depends on them.
# Anything in the dep closure that does NOT match this regex (gcc, libllvm,
# xserver-xorg-core, libdrm, ...) is a generic OS package and goes to debs/.
NVIDIA_PATTERN='^(nvidia-|libnvidia-|linux-modules-nvidia-|linux-objects-nvidia-|linux-signatures-nvidia-|xserver-xorg-video-nvidia-)'

# Compute the apt-rdepends closure for a list of seed packages and print
# package names (one per line, deduplicated) to stdout. Lines that don't look
# like bare names (alternatives, "<*** virtual ***>", etc.) are filtered out.
compute_closure() {
    local label="$1"; shift
    local pkgs=("$@")
    local out
    out=$(mktemp)
    for pkg in "${pkgs[@]}"; do
        if ! apt-rdepends "$pkg" 2>/dev/null \
                | grep -E '^[a-zA-Z0-9][a-zA-Z0-9.+:-]*$' \
                >> "$out"; then
            warn "[$label] apt-rdepends failed for $pkg (continuing)"
        fi
    done
    sort -u "$out"
    rm -f "$out"
}

# Look up the expected on-disk filename and size for the candidate version of
# a package without downloading anything. Echoes "<filename>\t<size>" or empty
# if the package isn't in apt's cache (virtual / unavailable).
#
# `apt-get download --print-uris` prints one line per package:
#   '<uri>' <filename> <size> [<hash-spec>]
apt_pkg_info() {
    local pkg="$1"
    apt-get download --print-uris "$pkg" 2>/dev/null \
        | awk '$2 ~ /\.deb$/ { print $2 "\t" $3; exit }'
}

# Download every package in a newline-separated list (from stdin) into a target
# directory using `apt-get download`. Idempotent: a file already on disk with
# the apt-cache-reported size is treated as fresh and skipped. Files that exist
# but with the wrong size (truncated download, version drift, partial transfer)
# are deleted and re-fetched. Virtual packages and unavailable names fail
# gracefully.
download_into() {
    local target="$1"
    local label="$2"
    log "[$label] downloading into $target (incremental, size-checked)"
    local total=0 cached=0 fetched=0 stale=0 failed=0
    pushd "$target" >/dev/null
    while read -r p; do
        [[ -z "$p" ]] && continue
        total=$((total+1))
        local meta exp_name exp_size on_disk
        meta=$(apt_pkg_info "$p")
        if [[ -n "$meta" ]]; then
            exp_name="${meta%%$'\t'*}"
            exp_size="${meta##*$'\t'}"
            if [[ -f "$exp_name" ]]; then
                on_disk=$(stat -c%s "$exp_name" 2>/dev/null || echo 0)
                if [[ "$on_disk" == "$exp_size" ]]; then
                    cached=$((cached+1))
                    continue
                fi
                warn "[$label] stale: $exp_name ($on_disk B vs expected $exp_size B) -- replacing"
                rm -f "$exp_name"
                stale=$((stale+1))
            fi
        fi
        if apt-get download "$p" >/dev/null 2>&1; then
            fetched=$((fetched+1))
        else
            failed=$((failed+1))
        fi
    done
    popd >/dev/null
    log "[$label] $total seeded -> $cached cached, $fetched fetched, $stale stale-replaced, $failed failed"
    local n size
    n=$(find "$target" -maxdepth 1 -name '*.deb' 2>/dev/null | wc -l)
    size=$(du -sh "$target" 2>/dev/null | cut -f1)
    log "[$label] dir now: $n files ($size)"
}

# ---------------- detect NVIDIA driver first (its closure feeds debs/) -------
NV_MAJOR=""
NV_PKG=""
if [[ "$SKIP_NVIDIA" != "1" ]]; then
    if [[ "$TARGET_DRIVER" == "auto" ]]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
            if [[ "$ver" =~ ^([0-9]+)\.([0-9]+) ]]; then
                NV_MAJOR="${BASH_REMATCH[1]}"
                log "[nvidia] auto-detected from nvidia-smi: $ver -> nvidia-driver-$NV_MAJOR"
            fi
        fi
        if [[ -z "$NV_MAJOR" ]]; then
            NV_MAJOR=$(dpkg -l 2>/dev/null | awk '/^ii.*nvidia-driver-[0-9]+/ {print $2}' \
                | sed 's/^nvidia-driver-//' | sort -V | tail -1)
            [[ -n "$NV_MAJOR" ]] && log "[nvidia] auto-detected from dpkg: nvidia-driver-$NV_MAJOR"
        fi
    else
        NV_MAJOR="$TARGET_DRIVER"
        log "[nvidia] using explicit TARGET_DRIVER=$NV_MAJOR"
    fi

    if [[ -n "$NV_MAJOR" ]]; then
        # Prefer the open-kernel-module variant; fall back to the proprietary one.
        NV_PKG="nvidia-driver-${NV_MAJOR}-open"
        if ! apt-cache show "$NV_PKG" >/dev/null 2>&1; then
            NV_PKG="nvidia-driver-${NV_MAJOR}"
        fi
    else
        warn "[nvidia] no driver detected; skipping. Set TARGET_DRIVER=NNN to force."
    fi
fi

# ---------------- ensure ppas + tools (used by both blocks) ----------------
if [[ "$SKIP_DEBS" != "1" || -n "$NV_PKG" ]]; then
    log "ensuring deadsnakes PPA + apt-rdepends"
    run "sudo apt-get update -y -qq"
    run "sudo apt-get install -y -qq software-properties-common ca-certificates curl gnupg apt-rdepends"
    run "sudo add-apt-repository -y ppa:deadsnakes/ppa"
    run "sudo apt-get update -y -qq"
fi

# ---------------- compute closures ----------------
NVIDIA_CLOSURE=""
NVIDIA_OWN=""        # subset matching $NVIDIA_PATTERN -> nvidia-debs/
NVIDIA_SUPPORT=""    # rest of the closure -> goes into debs/ (gcc, X server, ...)
if [[ -n "$NV_PKG" ]]; then
    log "[nvidia] computing closure for $NV_PKG"
    NVIDIA_CLOSURE=$(compute_closure "nvidia" "$NV_PKG")
    NVIDIA_OWN=$(echo "$NVIDIA_CLOSURE" | grep -E "$NVIDIA_PATTERN" || true)
    NVIDIA_SUPPORT=$(echo "$NVIDIA_CLOSURE" | grep -vE "$NVIDIA_PATTERN" || true)
    log "[nvidia] closure: $(echo "$NVIDIA_CLOSURE" | wc -l) total = $(echo "$NVIDIA_OWN" | wc -l) nvidia-specific + $(echo "$NVIDIA_SUPPORT" | wc -l) supporting (-> debs/)"
fi

# ---------------- generic DEBs (debs/) ----------------
if [[ "$SKIP_DEBS" != "1" ]]; then
    TOP_PKGS=(
        "ffmpeg"
        "python${PYTHON_VERSION}"
        "python${PYTHON_VERSION}-venv"
        "python${PYTHON_VERSION}-dev"
        "ca-certificates"
        "curl"
        "gnupg"
        "pciutils"
    )

    # Kernel headers for DKMS: nvidia-dkms compiles the kernel module against
    # the running kernel's headers on the offline target. Decide which kernel
    # to gather headers for.
    KERNEL_FOR_HEADERS=""
    case "$TARGET_KERNEL" in
        auto)   KERNEL_FOR_HEADERS=$(uname -r)
                log "[debs] kernel headers: TARGET_KERNEL=auto -> $(uname -r) (build host's kernel)" ;;
        "")     log "[debs] kernel headers: skipped (TARGET_KERNEL is empty)" ;;
        *)      KERNEL_FOR_HEADERS="$TARGET_KERNEL"
                log "[debs] kernel headers: TARGET_KERNEL=$TARGET_KERNEL (explicit)" ;;
    esac
    if [[ -n "$KERNEL_FOR_HEADERS" ]]; then
        # Two packages cover the chain:
        #   linux-headers-X.Y.Z-N-generic   <- kernel-version specific (full)
        #   linux-headers-X.Y.Z-N           <- abstract (shared headers)
        # Both must be at exactly the same version. We add the concrete one as
        # a top-level seed; apt-rdepends will pull in the abstract one.
        if apt-cache show "linux-headers-${KERNEL_FOR_HEADERS}" >/dev/null 2>&1; then
            TOP_PKGS+=("linux-headers-${KERNEL_FOR_HEADERS}")
        else
            warn "[debs] linux-headers-${KERNEL_FOR_HEADERS} not in apt cache on this host"
            warn "       (the offline target needs its kernel's headers; either run this build"
            warn "        on a host running that kernel, or pre-install on the target)"
        fi
    fi

    log "[debs] computing closure for: ${TOP_PKGS[*]}"
    DEBS_LIST=$(compute_closure "debs" "${TOP_PKGS[@]}")

    # Add the supporting packages from the nvidia closure (X server, gcc, ...)
    # so the offline target has them once, in debs/. Deduplicate.
    if [[ -n "$NVIDIA_SUPPORT" ]]; then
        DEBS_LIST=$(printf '%s\n%s\n' "$DEBS_LIST" "$NVIDIA_SUPPORT" | sort -u)
    fi
    log "[debs] $(echo "$DEBS_LIST" | wc -l) unique packages to fetch"

    # Don't wipe -- download_into now skips files with matching size. To force a
    # clean rebuild, delete debs/ and nvidia-debs/ manually before re-running.
    echo "$DEBS_LIST" | download_into "$DEBS_DIR" "debs"
else
    log "skipping debs (SKIP_DEBS=1)"
fi

# ---------------- NVIDIA driver DEBs (nvidia-debs/) ----------------
if [[ "$SKIP_NVIDIA" != "1" && -n "$NVIDIA_OWN" ]]; then
    log "[nvidia] downloading $(echo "$NVIDIA_OWN" | wc -l) NVIDIA-specific packages -> $NVIDIA_DEBS_DIR"
    echo "$NVIDIA_OWN" | download_into "$NVIDIA_DEBS_DIR" "nvidia"
elif [[ "$SKIP_NVIDIA" == "1" ]]; then
    log "skipping nvidia debs (SKIP_NVIDIA=1)"
fi

# ---------------- pip wheels ----------------
if [[ "$SKIP_PIP" != "1" ]]; then
    PY="python${PYTHON_VERSION}"
    command -v "$PY" >/dev/null || die "$PY not on PATH (needed to resolve wheels)"

    TMPVENV="$OFFLINE_DIR/.tmp-resolve-venv"
    log "creating temporary venv at $TMPVENV (used only to resolve dep versions)"
    run "rm -rf '$TMPVENV'"
    run "$PY -m venv '$TMPVENV'"
    run "'$TMPVENV/bin/pip' install --upgrade -q pip wheel setuptools"

    # Step 1: install torch+torchaudio from CUDA-specific index so freeze records
    # the right +cuXXX local version specifier.
    log "installing torch ($TORCH_CUDA) into resolver venv"
    run "'$TMPVENV/bin/pip' install -q --index-url https://download.pytorch.org/whl/$TORCH_CUDA torch torchaudio"

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
else
    log "skipping pip (SKIP_PIP=1)"
fi

# ---------------- manifest ----------------
log "writing manifest -> $MANIFEST"
{
    echo "# video_ai_subtitle offline-packages bundle"
    echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host:      $(uname -a)"
    echo "# os:        ${PRETTY_NAME:-?}"
    echo "# nvidia driver detected: ${NV_MAJOR:-(none)}"
    echo "# python:        $PYTHON_VERSION"
    echo "# torch_cuda:    $TORCH_CUDA"
    echo "# pip_extras:    $PIP_EXTRAS"
    echo
    echo "## nvidia-debs/  (NVIDIA driver, install before debs/)"
    (cd "$NVIDIA_DEBS_DIR" && ls -1 *.deb 2>/dev/null) || echo "(none)"
    echo
    echo "## debs/"
    (cd "$DEBS_DIR" && ls -1 *.deb 2>/dev/null) || echo "(none)"
    echo
    echo "## pip-wheels/"
    (cd "$WHEELS_DIR" && ls -1 2>/dev/null) || echo "(none)"
} > "$MANIFEST"

# ---------------- integrity checksums ----------------
SUMS_FILE="$OFFLINE_DIR/sha256sums.txt"
log "writing checksums -> $SUMS_FILE"
if [[ "$DRY_RUN" != "1" ]]; then
    (
        cd "$OFFLINE_DIR"
        # Hash every artifact file (debs, wheels, requirements.txt, manifest).
        # `find -print0 | sort -z` keeps ordering deterministic across runs and
        # safe against filenames with spaces / special chars.
        find debs nvidia-debs pip-wheels manifest.txt \
            -type f \
            \( -name '*.deb' -o -name '*.whl' -o -name '*.tar.gz' \
               -o -name 'requirements.txt' -o -name 'manifest.txt' \) \
            -print0 2>/dev/null \
            | sort -z \
            | xargs -0 sha256sum > "$SUMS_FILE"
    )
    log "checksums: $(wc -l < "$SUMS_FILE") files, $(du -h "$SUMS_FILE" | cut -f1)"
fi

log "DONE."
log "next:"
log "  bash offline_packages/build_models.sh                    # download model weights"
log ""
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
