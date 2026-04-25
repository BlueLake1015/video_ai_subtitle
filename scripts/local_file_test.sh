#!/usr/bin/env bash
# End-to-end smoke test against the local fixture.
#
# Default:  one transcribe case (medium) + one translate case (medium + fast).
# --full:   run the whole preset matrix.
#
# Each `vas` invocation is echoed before it runs, so you can copy/paste any
# failing command and re-run it directly for debugging.
#
# Requirements:
#   * .venv created and project pip-installed (`bash scripts/install_profile_a.sh`)
#   * NVIDIA driver loaded (`nvidia-smi` must work; reboot after driver install)
#   * Translation cases additionally need:
#       - HF_TOKEN set, or `huggingface-cli login` done
#       - Gemma license accepted on huggingface.co for the chosen model
#     If neither is present, translation cases are skipped automatically.
#
# Usage:
#   bash scripts/local_file_test.sh             # default (1 transcribe + 1 translate)
#   bash scripts/local_file_test.sh --full      # full matrix
#   FIXTURE=in.mp4 bash scripts/local_file_test.sh --full
#   SRC_LANG=ja TGT_LANG=en bash scripts/local_file_test.sh

set -uo pipefail

# ---------------- argv ----------------
MODE="default"
for arg in "$@"; do
    case "$arg" in
        --full)    MODE="full" ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *)         echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# ---------------- tunables ----------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
FIXTURE="${FIXTURE:-$PROJECT_DIR/tests/fixtures/test_en_30s.mp4}"
OUT_DIR="${OUT_DIR:-/tmp/vas-smoke}"
SRC_LANG="${SRC_LANG:-en}"
TGT_LANG="${TGT_LANG:-ko}"
KEEP_OUTPUTS="${KEEP_OUTPUTS:-1}"  # 0 = wipe OUT_DIR before run

# ---------------- preset sets ----------------
# Default mode: a single sensible case each.
DEFAULT_TRANSCRIBE_CASES=("medium")
DEFAULT_TRANSLATE_CASES=("medium|fast")   # transcribe-preset|translate-preset

# Full matrix.
FULL_TRANSCRIBE_CASES=(
    "tiny"
    "base"
    "small"
    "medium"
    "large-v3-turbo"
    "distil-large-v3"
    "quality"
)
FULL_TRANSLATE_CASES=(
    "tiny|edge"
    "medium|fast"
    "large-v3-turbo|balanced"
)

if [[ "$MODE" == "full" ]]; then
    TRANSCRIBE_CASES=("${FULL_TRANSCRIBE_CASES[@]}")
    TRANSLATE_CASES=("${FULL_TRANSLATE_CASES[@]}")
else
    TRANSCRIBE_CASES=("${DEFAULT_TRANSCRIBE_CASES[@]}")
    TRANSLATE_CASES=("${DEFAULT_TRANSLATE_CASES[@]}")
fi

MIN_CUES=1
FORMATS=(srt ttml)

# ---------------- helpers ----------------
log()  { printf '\033[1;36m[test ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[FAIL ]\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m[ ok  ]\033[0m %s\n' "$*"; }
cmd()  { printf '\033[2m+ %s\033[0m\n' "$*"; }       # dim grey for command echo
hr()   { printf -- '----------------------------------------\n'; }

# Quote each argument so the printed command is copy-paste safe.
printf_cmd() {
    local out=""
    for a in "$@"; do
        if [[ "$a" =~ [^A-Za-z0-9_./:=-] ]]; then
            out+=" '${a//\'/\'\\\'\'}'"
        else
            out+=" $a"
        fi
    done
    cmd "${out# }"
}

# ---------------- preflight ----------------
log "mode:        $MODE"
log "project dir: $PROJECT_DIR"
log "fixture:     $FIXTURE"
log "out dir:     $OUT_DIR"
log "language:    $SRC_LANG -> $TGT_LANG"

if [[ ! -d "$VENV_DIR" ]]; then
    fail "venv not found at $VENV_DIR -- run scripts/install_profile_a.sh first"
    exit 1
fi
VAS="$VENV_DIR/bin/vas"
if [[ ! -x "$VAS" ]]; then
    fail "$VAS not executable -- did pip install -e succeed?"
    exit 1
fi
if [[ ! -f "$FIXTURE" ]]; then
    fail "fixture missing: $FIXTURE  (see tests/fixtures/README.md to regenerate)"
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not working; transcribe runs may fall back to CPU or fail"
fi

mkdir -p "$OUT_DIR"
[[ "$KEEP_OUTPUTS" == "0" ]] && rm -f "$OUT_DIR"/*.srt "$OUT_DIR"/*.ttml "$OUT_DIR"/*.log

# Detect HF auth (needed for Gemma translation).
HF_AUTHED=0
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_AUTHED=1
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    HF_AUTHED=1
fi
if [[ "$HF_AUTHED" == "0" ]]; then
    warn "no HF_TOKEN / huggingface-cli login -- translation cases will be skipped"
    warn "see README section 'HF_TOKEN' for how to enable them"
fi

# ---------------- collect results ----------------
declare -a RESULTS=()    # each entry: "STATUS|NAME|SECONDS|CUES|NOTES"

run_case() {
    local name="$1"; shift
    local out="$1"; shift
    local logf="$1"; shift
    local min_cues="$1"; shift
    # remaining args = vas subtitle args

    log "running: $name"
    printf_cmd "$VAS" subtitle "$@"

    local t0 t1 elapsed
    t0=$(date +%s.%N)
    if "$VAS" subtitle "$@" >"$logf" 2>&1; then
        t1=$(date +%s.%N)
        elapsed=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
        local cues=0
        if [[ -f "$out" ]]; then
            if [[ "$out" == *.srt ]]; then
                cues=$(grep -cE '^[0-9]+$' "$out" || echo 0)
            elif [[ "$out" == *.ttml ]]; then
                cues=$(grep -cE '<p ' "$out" || echo 0)
            fi
        fi
        if (( cues < min_cues )); then
            fail "$name: produced only $cues cues (expected >= $min_cues)"
            RESULTS+=("FAIL|$name|${elapsed}s|$cues|< min_cues")
            return 1
        fi
        ok "$name: ${elapsed}s, $cues cues -> $out"
        RESULTS+=("ok|$name|${elapsed}s|$cues|")
        return 0
    else
        t1=$(date +%s.%N)
        elapsed=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
        local last_line
        last_line=$(tail -3 "$logf" | tr '\n' ' ' | cut -c1-100)
        fail "$name: failed after ${elapsed}s -- $last_line"
        RESULTS+=("FAIL|$name|${elapsed}s|0|see $logf")
        return 1
    fi
}

# ---------------- transcription cases ----------------
hr
log "== transcription only =="
hr

for preset in "${TRANSCRIBE_CASES[@]}"; do
    fmt="${FORMATS[0]}"
    out="$OUT_DIR/asr_${preset}.${fmt}"
    logf="$OUT_DIR/asr_${preset}.${fmt}.log"
    run_case "asr/${preset}/${fmt}" "$out" "$logf" "$MIN_CUES" \
        "$FIXTURE" -o "$out" -t "$preset" --src-lang "$SRC_LANG"
done

# ---------------- translation cases ----------------
if [[ "$HF_AUTHED" == "1" ]]; then
    hr
    log "== transcription + translation ($SRC_LANG -> $TGT_LANG) =="
    hr

    for case in "${TRANSLATE_CASES[@]}"; do
        IFS="|" read -r tpreset mpreset <<< "$case"
        out="$OUT_DIR/mt_${tpreset}_${mpreset}.srt"
        logf="$OUT_DIR/mt_${tpreset}_${mpreset}.log"
        run_case "asr+mt/${tpreset}+${mpreset}" "$out" "$logf" "$MIN_CUES" \
            "$FIXTURE" -o "$out" \
            -t "$tpreset" -T "$mpreset" \
            --src-lang "$SRC_LANG" --tgt-lang "$TGT_LANG"
    done
else
    log "(skipping translation cases: no HF auth)"
fi

# ---------------- summary ----------------
hr
log "summary"
hr
printf "%-6s %-40s %-10s %-6s %s\n" "STATUS" "CASE" "TIME" "CUES" "NOTES"
for r in "${RESULTS[@]}"; do
    IFS="|" read -r status name secs cues notes <<< "$r"
    if [[ "$status" == "ok" ]]; then
        printf "\033[1;32m%-6s\033[0m %-40s %-10s %-6s %s\n" "PASS" "$name" "$secs" "$cues" "$notes"
    else
        printf "\033[1;31m%-6s\033[0m %-40s %-10s %-6s %s\n" "$status" "$name" "$secs" "$cues" "$notes"
    fi
done
hr
log "outputs in $OUT_DIR (set KEEP_OUTPUTS=0 to wipe each run)"

# Exit non-zero if anything failed.
fail_count=0
for r in "${RESULTS[@]}"; do
    [[ "${r%%|*}" == "FAIL" ]] && fail_count=$((fail_count + 1))
done
if (( fail_count > 0 )); then
    fail "$fail_count case(s) failed"
    exit 1
fi
ok "all ${#RESULTS[@]} cases passed"
