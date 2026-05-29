#!/usr/bin/env bash
# Gather every .deb (apt packages, transitive closure) needed to install the
# project on an air-gapped Ubuntu target. Run on a machine that DOES have
# internet, ideally same Ubuntu version + GPU class as the offline target.
#
# This is the APT half of the bundle. For Python wheels run
# `download_2_python_packages.sh`. Produces:
#
#   offline_packages/
#     debs/                  *.deb   (generic OS packages: ffmpeg, python, headers, …)
#     nvidia-debs/           *.deb   (NVIDIA driver, install before debs/)
#     manifest_ubuntu.txt    inventory of the apt artifacts
#
# Usage:
#   bash offline_packages/download_1_ubuntu_packages.sh
#
# Knobs (env vars):
#   TARGET_DRIVER=560        nvidia-driver major to gather (default auto-detect)
#   PYTHON_VERSION=3.11      python series debs to gather  (default 3.11)
#   TARGET_KERNEL=auto       kernel for DKMS headers: auto | "" (skip) | X.Y.Z-N-generic
#   OFFLINE_DIR=path/        output directory (default $PROJECT_DIR/offline_packages)
#   SKIP_DEBS=1              don't gather generic .debs
#   SKIP_NVIDIA=1            don't gather nvidia-driver .debs
#   DRY_RUN=1                preview, don't execute

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_DIR="${OFFLINE_DIR:-$PROJECT_DIR/offline_packages}"

# Shared build profile (target GPU / CUDA / torch / python). Values there are
# assign-if-unset, so explicit `VAR=... bash <script>` overrides still win.
TARGET_ENV="${TARGET_ENV:-$PROJECT_DIR/offline_packages/target.env}"
[[ -f "$TARGET_ENV" ]] && source "$TARGET_ENV"

# NVIDIA driver major. Normally set by target.env (570 for the RTX 4090 target).
# If target.env is absent, falls back to "auto" = detect this build host's driver
# via nvidia-smi (only correct when build host == target). Override per-run with
# TARGET_DRIVER=NNN.
TARGET_DRIVER="${TARGET_DRIVER:-auto}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
SKIP_DEBS="${SKIP_DEBS:-0}"
SKIP_NVIDIA="${SKIP_NVIDIA:-0}"   # 1 = don't gather nvidia-driver debs
DRY_RUN="${DRY_RUN:-0}"
# Target kernel for DKMS (nvidia-dkms-XXX-open builds the kernel module against
# linux-headers-$(uname -r) at install time on the offline host).
#   auto     -> use this build host's kernel (only useful if target == build host)
#   <empty>  -> don't gather kernel headers (target must already have them)
#   5.15.0-176-generic -> explicit kernel version of the offline target
TARGET_KERNEL="${TARGET_KERNEL:-auto}"

log()  { printf '\033[1;36m[ubuntu]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn  ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[err   ]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then printf '\033[2m+ %s\033[0m\n' "$*"; else eval "$@"; fi; }

[[ -r /etc/os-release ]] || die "/etc/os-release missing -- this script targets Ubuntu"
. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "unsupported OS: ${ID:-unknown}"

mkdir -p "$OFFLINE_DIR"
DEBS_DIR="$OFFLINE_DIR/debs"
NVIDIA_DEBS_DIR="$OFFLINE_DIR/nvidia-debs"
MANIFEST="$OFFLINE_DIR/manifest_ubuntu.txt"
mkdir -p "$DEBS_DIR" "$NVIDIA_DEBS_DIR"

log "project:        $PROJECT_DIR"
log "offline dir:    $OFFLINE_DIR"
log "target driver:  $TARGET_DRIVER"
log "python series:  $PYTHON_VERSION"

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

# ---------------- ensure ppas + tools ----------------
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

# ---------------- manifest (apt artifacts) ----------------
log "writing manifest -> $MANIFEST"
{
    echo "# video_ai_subtitle offline-packages bundle (apt half)"
    echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host:      $(uname -a)"
    echo "# os:        ${PRETTY_NAME:-?}"
    echo "# nvidia driver detected: ${NV_MAJOR:-(none)}"
    echo "# python:        $PYTHON_VERSION"
    echo
    echo "## nvidia-debs/  (NVIDIA driver, install before debs/)"
    (cd "$NVIDIA_DEBS_DIR" && ls -1 *.deb 2>/dev/null) || echo "(none)"
    echo
    echo "## debs/"
    (cd "$DEBS_DIR" && ls -1 *.deb 2>/dev/null) || echo "(none)"
} > "$MANIFEST"

log "DONE (apt half)."
log "next:"
log "  bash offline_packages/download_2_python_packages.sh    # gather pip wheels"
log "  bash offline_packages/download_3_models.sh             # download model weights"
