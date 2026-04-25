#!/usr/bin/env bash
# Re-stream a local media file as MPEG2-TS to a loopback address, so the live
# pipeline (`vas subtitle udp://127.0.0.1:5000 ...`) can be exercised without
# a real broadcast feed.
#
# Default mode: MPEG2-TS over UDP -- the most common IPTV/contribution pattern.
# --rtp:        MPEG2-TS encapsulated in RTP (RFC 2250); produces an SDP file.
#
# Usage:
#   bash scripts/stream_local_file.sh [INPUT] [--host HOST] [--port PORT] [--rtp] [--loop]
#
# Defaults:
#   INPUT  = tests/fixtures/test_en_30s.mp4
#   HOST   = 127.0.0.1
#   PORT   = 5000
#
# Two-terminal recipe (UDP TS):
#   1. terminal A:   vas subtitle udp://127.0.0.1:5000 -o /tmp/live.srt -t medium --src-lang en
#   2. terminal B:   bash scripts/stream_local_file.sh                    # starts streaming
#
# Always start the receiver first; UDP has no connection setup, so packets sent
# before the receiver is ready are silently dropped.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$PROJECT_DIR/tests/fixtures/test_en_30s.mp4"
HOST="127.0.0.1"
PORT="5000"
PROTO="udp"          # udp | rtp
LOOP="0"             # 0 | 1 (-stream_loop -1)

# ---------------- argv ----------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --rtp)  PROTO="rtp"; shift ;;
        --loop) LOOP="1"; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        -*) echo "unknown flag: $1" >&2; exit 2 ;;
        *)  INPUT="$1"; shift ;;
    esac
done

if [[ ! -f "$INPUT" ]]; then
    echo "input file not found: $INPUT" >&2
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found on PATH" >&2
    exit 1
fi

# ---------------- ffmpeg invocation ----------------
echo "[stream] input:  $INPUT"
echo "[stream] target: ${PROTO}://${HOST}:${PORT}"
[[ "$LOOP" == "1" ]] && echo "[stream] looping: yes (Ctrl+C to stop)"

LOOP_FLAG=()
[[ "$LOOP" == "1" ]] && LOOP_FLAG=(-stream_loop -1)

if [[ "$PROTO" == "rtp" ]]; then
    SDP_FILE="/tmp/vas_stream_${PORT}.sdp"
    echo "[stream] sdp:    $SDP_FILE"
    echo
    echo "Receiver command (in another terminal):"
    echo "  vas subtitle 'rtp://${HOST}:${PORT}' -o /tmp/live.srt -t medium --src-lang en"
    echo
    exec ffmpeg -hide_banner -loglevel info -re "${LOOP_FLAG[@]}" \
        -i "$INPUT" \
        -c copy \
        -f rtp_mpegts \
        -sdp_file "$SDP_FILE" \
        "rtp://${HOST}:${PORT}"
else
    echo
    echo "Receiver command (in another terminal):"
    echo "  vas subtitle 'udp://${HOST}:${PORT}' -o /tmp/live.srt -t medium --src-lang en"
    echo
    # pkt_size=1316 = 7 * 188-byte TS packets per UDP datagram (standard MTU-safe).
    exec ffmpeg -hide_banner -loglevel info -re "${LOOP_FLAG[@]}" \
        -i "$INPUT" \
        -c copy \
        -f mpegts \
        "udp://${HOST}:${PORT}?pkt_size=1316"
fi
