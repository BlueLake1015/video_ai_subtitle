#!/usr/bin/env python3
"""Score a transcription against a reference and flag likely ASR hallucinations.

Built for the TED-LIUM ground truth in this directory
(`tedlium_5talks_en_reference.txt`), whose text is normalized TED-LIUM style:
lowercase, no punctuation, spaced apostrophes ("i 'm"), occasional <unk> tokens.
Both hypothesis and reference are normalized the same way before scoring, so
casing/punctuation differences don't inflate the WER.

It reports two things:

  1. WER (+ substitution / deletion / insertion breakdown when jiwer is present).
  2. Hallucination checks — the failure modes Whisper-family ASR shows on silence,
     music, and long talks:
       * repetition loops    — an n-gram repeated back-to-back many times
                              ("you know you know you know …")
       * duplicate cues      — the same cue text over CONSECUTIVE cues
       * repeated cues       — the same cue text recurring many times SCATTERED
                              through the file (loops/dupes only catch adjacent)
       * phantom phrases     — standalone cues that are known idle-output phrases
                              ("thank you", "oh my god", "i love you", …)
       * insertion clusters  — long runs of hypothesis words with no reference
                              match (from the alignment; needs jiwer)

Usage:
    # WER + hallucination checks (with a reference):
    python tests/data/score_wer.py \
        --hyp /tmp/ted.en.srt \
        --ref tests/data/tedlium_5talks_en_reference.txt

    # Hallucination-only (no ground truth, e.g. the film): omit --ref. WER and the
    # insertion-cluster check are skipped; loops / duplicate / repeated / phantom
    # checks still run.
    python tests/data/score_wer.py --hyp /tmp/notld.en.srt

The hypothesis may be .srt, .vtt, or plain .txt. SRT/VTT markup and timestamps
are parsed automatically; with timestamps present, flagged spans are reported at
their wall-clock position.

Uses `jiwer` if installed (fast, enables insertion-cluster detection); otherwise
falls back to a slower built-in WER and skips the alignment-based check.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Known idle-output / boilerplate phrases Whisper emits on non-speech. Compared
# against a whole cue (already normalized: lowercase, no punctuation, apostrophes
# rejoined so "don't" stays "don't"). Heuristic — many of these ("thank you",
# "you", "oh my god") also occur in real speech, so they are reported as
# candidates to eyeball, not hard errors.
PHANTOM_PHRASES = {
    # idle / filler
    "you",
    "bye",
    "bye bye",
    "goodbye",
    "the end",
    "oh my god",
    "oh my gosh",
    "i love you",
    "i love you guys",
    # gratitude / closings (very common Whisper hallucination on trailing silence)
    "thank you",
    "thanks",
    "thank you very much",
    "thank you so much",
    "thank you all",
    "thank you all so much",
    "thanks for watching",
    "thanks for watching everyone",
    "thank you for watching",
    # YouTube-training-data artifacts
    "please subscribe",
    "please like and subscribe",
    "like and subscribe",
    "like comment and subscribe",
    "don't forget to subscribe",
    "subscribe to my channel",
    "hit the like button",
    "see you next time",
    "see you in the next video",
    "i'll see you next time",
    "i'll see you in the next video",
    # non-speech sound tags that survive normalization
    "music",
    "music playing",
    "upbeat music",
    "intro music",
    "outro music",
    "instrumental music",
    "applause",
    "laughter",
    # transcription credits
    "subtitles by the amara org community",
    "transcription by castingwords",
    "transcribed by",
}


@dataclass
class Cue:
    start: float | None  # seconds, None for plain-text input
    text: str


def parse_cues(text: str, is_subtitle: bool) -> list[Cue]:
    """Parse SRT/VTT into timed cues, or wrap plain text as a single untimed cue."""
    if not is_subtitle:
        return [Cue(None, text)]
    cues: list[Cue] = []
    cur_start: float | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        if cur_lines:
            cues.append(Cue(cur_start, " ".join(cur_lines)))

    for line in text.splitlines():
        s = line.strip()
        if not s or s.upper().startswith("WEBVTT"):
            continue
        if "-->" in s:
            flush()
            cur_lines = []
            m = re.search(r"(\d+):(\d+):(\d+)[.,](\d+)", s)
            if m:
                h, mi, sec, ms = (int(g) for g in m.groups())
                cur_start = h * 3600 + mi * 60 + sec + ms / 1000.0
            else:
                cur_start = None
            continue
        if re.fullmatch(r"\d+", s):  # cue index
            continue
        cur_lines.append(s)
    flush()
    return cues


def normalize(text: str) -> list[str]:
    """Normalize to TED-LIUM-style tokens.

    Lowercase, drop <unk>, rejoin spaced apostrophes so "i 'm" and "i'm" match,
    remove remaining punctuation, split on whitespace.
    """
    t = text.lower()
    t = t.replace("<unk>", " ")
    t = re.sub(r"\s+'", "'", t)      # "i 'm" -> "i'm"
    t = re.sub(r"[^\w\s']", " ", t)  # drop punctuation, keep word chars + apostrophe
    t = t.replace("_", " ")          # \w keeps underscore; drop it
    return t.split()


def fmt_time(t: float | None) -> str:
    if t is None:
        return "--:--:--"
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# WER
# --------------------------------------------------------------------------- #
def wer_builtin(ref: list[str], hyp: list[str]) -> float:
    """Word-level Levenshtein WER, O(n*m) time / O(m) memory. Slower than jiwer."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return float(m > 0)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ref_word = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_word == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


# --------------------------------------------------------------------------- #
# Hallucination detectors
# --------------------------------------------------------------------------- #
def find_repetition_loops(tokens: list[str], max_n: int, min_repeats: int):
    """Find n-grams (1..max_n) repeated >= min_repeats times consecutively.

    Returns (hyp_token_index, phrase_tokens, repeat_count), non-overlapping,
    preferring the longest covered span at each position.
    """
    loops = []
    i, N = 0, len(tokens)
    while i < N:
        best = None  # (n, reps, span)
        for n in range(1, max_n + 1):
            if i + n > N:
                break
            gram = tokens[i : i + n]
            reps, j = 1, i + n
            while j + n <= N and tokens[j : j + n] == gram:
                reps += 1
                j += n
            if reps >= min_repeats and (best is None or reps * n > best[2]):
                best = (n, reps, reps * n)
        if best:
            n, reps, span = best
            loops.append((i, tokens[i : i + n], reps))
            i += span
        else:
            i += 1
    return loops


def find_duplicate_cue_runs(cues: list[Cue], min_run: int):
    """Find runs of >= min_run consecutive cues with identical normalized text.

    Returns (first_cue_index, start_time, text, count).
    """
    runs = []
    norm = [(" ".join(normalize(c.text)), c.start) for c in cues]
    i = 0
    while i < len(norm):
        text, start = norm[i]
        j = i + 1
        while j < len(norm) and norm[j][0] == text:
            j += 1
        if text and (j - i) >= min_run:
            runs.append((i, start, text, j - i))
        i = j
    return runs


def find_phantom_cues(cues: list[Cue]):
    """Returns (cue_index, start_time, normalized_text) for phantom-phrase cues."""
    out = []
    for k, c in enumerate(cues):
        norm = " ".join(normalize(c.text))
        if norm in PHANTOM_PHRASES:
            out.append((k, c.start, norm))
    return out


def find_repeated_cues(cues: list[Cue], min_count: int, min_words: int):
    """Find cue text that recurs many times across the whole file, even when the
    occurrences are NOT consecutive (the loop/duplicate detectors only catch
    back-to-back repeats). Whisper sometimes emits the same hallucinated line
    over and over, scattered through a long transcript.

    Returns (text, count, cue_indices) sorted by descending count.
    """
    norm = [" ".join(normalize(c.text)) for c in cues]
    index_of: dict[str, list[int]] = {}
    for k, t in enumerate(norm):
        if t and len(t.split()) >= min_words:
            index_of.setdefault(t, []).append(k)
    out = [(t, len(idxs), idxs) for t, idxs in index_of.items() if len(idxs) >= min_count]
    out.sort(key=lambda x: -x[1])
    return out


def find_insertion_clusters(jiwer_output, hyp_tokens, hyp_times, min_run: int):
    """Runs of >= min_run consecutive inserted hyp words (no reference match)."""
    clusters = []
    for sentence in jiwer_output.alignments:
        for chunk in sentence:
            if chunk.type == "insert" and (chunk.hyp_end_idx - chunk.hyp_start_idx) >= min_run:
                span = hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx]
                start = hyp_times[chunk.hyp_start_idx] if chunk.hyp_start_idx < len(hyp_times) else None
                clusters.append((start, chunk.hyp_start_idx, span))
    return clusters


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hyp", required=True, type=Path, help="pipeline output: .srt/.vtt/.txt")
    ap.add_argument("--ref", type=Path, default=None,
                    help="reference transcript .txt; omit for hallucination-only mode "
                         "(no WER / insertion-cluster check)")
    ap.add_argument("--loop-ngram", type=int, default=4, help="max n-gram size for loop detection")
    ap.add_argument("--loop-repeats", type=int, default=4, help="min consecutive repeats to flag a loop")
    ap.add_argument("--dup-cues", type=int, default=3, help="min identical consecutive cues to flag")
    ap.add_argument("--repeat-cues", type=int, default=5, help="min total (scattered) occurrences of a cue to flag")
    ap.add_argument("--repeat-min-words", type=int, default=2, help="min words a cue must have to count for --repeat-cues")
    ap.add_argument("--insertion-run", type=int, default=8, help="min consecutive inserted words to flag")
    args = ap.parse_args()

    if not args.hyp.exists():
        sys.exit(f"hypothesis not found: {args.hyp}")
    if args.ref is not None and not args.ref.exists():
        sys.exit(f"reference not found: {args.ref}")

    is_sub = args.hyp.suffix.lower() in (".srt", ".vtt")
    cues = parse_cues(args.hyp.read_text(encoding="utf-8", errors="replace"), is_sub)

    # Build the hypothesis token stream with a parallel per-token timestamp list,
    # so flagged spans can be reported at their wall-clock position.
    hyp_tokens: list[str] = []
    hyp_times: list[float | None] = []
    cue_spans: list[tuple[int, int]] = []  # (tok_start, tok_end) per cue
    for c in cues:
        toks = normalize(c.text)
        start = len(hyp_tokens)
        hyp_tokens.extend(toks)
        hyp_times.extend([c.start] * len(toks))
        cue_spans.append((start, len(hyp_tokens)))

    # ---- WER (only with a reference) ----
    jiwer_output = None
    if args.ref is None:
        print("=== WER ===")
        print(f"hypothesis words: {len(hyp_tokens)}")
        print("skipped: no --ref given (hallucination-only mode); "
              "insertion-cluster check also needs a reference")
    else:
        ref_tokens = normalize(args.ref.read_text(encoding="utf-8", errors="replace"))
        if not ref_tokens:
            sys.exit("reference is empty after normalization")
        try:
            import jiwer

            jiwer_output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
            wer = jiwer_output.wer
            backend = "jiwer"
        except ImportError:
            print(
                "note: jiwer not installed; using slower built-in WER and skipping the "
                "insertion-cluster check (pip install jiwer to enable both)",
                file=sys.stderr,
            )
            wer = wer_builtin(ref_tokens, hyp_tokens)
            backend = "builtin"

        print("=== WER ===")
        print(f"reference words : {len(ref_tokens)}")
        print(f"hypothesis words: {len(hyp_tokens)}")
        print(f"WER ({backend}) : {wer:.4f}  ({wer * 100:.2f}%)")
        if jiwer_output is not None:
            print(
                f"  substitutions={jiwer_output.substitutions} "
                f"deletions={jiwer_output.deletions} "
                f"insertions={jiwer_output.insertions} hits={jiwer_output.hits}"
            )

    # ---- Hallucination checks ----
    loops = find_repetition_loops(hyp_tokens, args.loop_ngram, args.loop_repeats)
    dups = find_duplicate_cue_runs(cues, args.dup_cues) if is_sub else []
    repeats = find_repeated_cues(cues, args.repeat_cues, args.repeat_min_words) if is_sub else []
    phantoms = find_phantom_cues(cues) if is_sub else []
    clusters = (
        find_insertion_clusters(jiwer_output, hyp_tokens, hyp_times, args.insertion_run)
        if jiwer_output is not None
        else []
    )

    print("\n=== Hallucination checks ===")
    flagged: set[int] = set()  # unique hyp token indices, so detectors don't double-count

    print(f"[repetition loops]   {len(loops)} found"
          f" (n-gram repeated >= {args.loop_repeats}x)")
    for idx, phrase, reps in loops[:20]:
        flagged.update(range(idx, idx + len(phrase) * reps))
        print(f"  {fmt_time(hyp_times[idx])}  \"{' '.join(phrase)}\" x{reps}")
    if len(loops) > 20:
        print(f"  … and {len(loops) - 20} more")

    print(f"[duplicate cues]     {len(dups)} run(s) (>= {args.dup_cues} consecutive identical cues)")
    for cue_idx, start, text, count in dups[:20]:
        flagged.update(range(cue_spans[cue_idx][0], cue_spans[min(cue_idx + count - 1, len(cue_spans) - 1)][1]))
        print(f"  {fmt_time(start)}  \"{text}\" x{count} cues")

    print(f"[repeated cues]      {len(repeats)} text(s) recur >= {args.repeat_cues}x (scattered)")
    for text, count, idxs in repeats[:20]:
        for ci in idxs:
            flagged.update(range(*cue_spans[ci]))
        preview = text if len(text) <= 70 else text[:70] + "…"
        print(f"  {fmt_time(cues[idxs[0]].start)}  \"{preview}\" x{count} cues")
    if len(repeats) > 20:
        print(f"  … and {len(repeats) - 20} more")

    print(f"[phantom phrases]    {len(phantoms)} candidate cue(s)")
    seen: dict[str, int] = {}
    for cue_idx, start, text in phantoms:
        flagged.update(range(*cue_spans[cue_idx]))
        seen[text] = seen.get(text, 0) + 1
    for text, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  \"{text}\" x{count}")

    if jiwer_output is not None:
        ins_note = ""
    elif args.ref is None:
        ins_note = "  (skipped: needs --ref)"
    else:
        ins_note = "  (skipped: needs jiwer)"
    print(f"[insertion clusters] {len(clusters)} run(s) (>= {args.insertion_run} inserted words){ins_note}")
    for start, idx, span in clusters[:20]:
        flagged.update(range(idx, idx + len(span)))
        preview = " ".join(span)
        if len(preview) > 90:
            preview = preview[:90] + "…"
        print(f"  {fmt_time(start)}  hyp[{idx}]: \"{preview}\"")
    if len(clusters) > 20:
        print(f"  … and {len(clusters) - 20} more")

    pct = 100.0 * len(flagged) / max(1, len(hyp_tokens))
    total = len(loops) + len(dups) + len(repeats) + len(phantoms) + len(clusters)
    print(f"\nsummary: {total} flag(s); {len(flagged)} distinct hypothesis words "
          f"({pct:.2f}%) in flagged spans")


if __name__ == "__main__":
    main()
