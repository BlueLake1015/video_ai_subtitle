# Long-form real-world test media

Large (>1 hour) real-world clips for stress-testing the full
`ffmpeg -> VAD -> segmenter -> Whisper -> (Gemma) -> SRT/TTML/VTT` pipeline.

This directory is **gitignored** (see `.gitignore: tests/data/`) — the media is
too large to track. Regenerate it with the download commands below.

Two complementary sets:

| File | Tests | Length | Ground truth |
|---|---|---|---|
| `nightofthelivingdead_1968_en.mp4` | **Robustness** — survives real film audio | 95.9 min | none (see note) |
| `tedlium_5talks_en.m4a` (+ `_reference.txt`) | **Accuracy (WER)** — real broadcast speech | 95.9 min | verbatim human transcript |

Why two: the film proves the pipeline *survives* messy real-world film audio for
90+ minutes; the TED-LIUM set lets you *measure* transcription accuracy against a
trustworthy reference. Neither is clean read-aloud audio (LibriVox-style), which
is training-grade material, not a real-world test.

---

## nightofthelivingdead_1968_en.mp4 — robustness

A full-length public-domain feature film: dialogue-heavy, multiple speakers,
1968 broadcast-era mono audio.

- **Title**: *Night of the Living Dead* (1968), dir. George A. Romero
- **Why public domain**: the original theatrical prints omitted a copyright
  notice, so the film entered the U.S. public domain on release.
- **Origin**: https://archive.org/details/Night.Of.The.Living.Dead_1080p
- **License**: Public Domain (`http://creativecommons.org/licenses/publicdomain/`)
- **Encoded (committed here)**: **audio-only** AAC stereo 44.1 kHz 128 kbit,
  ~95.9 min, ~95 MB. The video track was stripped (`ffmpeg -vn -c:a copy`) since
  the pipeline only consumes audio — this keeps the repo file under GitHub's size
  limit. The original Archive download is 640×480 h.264 + this audio, ~570 MB (the
  "1080p" label there is a misnomer; resolution is irrelevant for ASR).

### Test commands
Run from the repo root with `.venv` active (or via `make run …`).

```bash
# Faster smoke run first — confirm the pipeline gets through the whole film:
vas subtitle tests/data/nightofthelivingdead_1968_en.mp4 \
    -o /tmp/notld.fast.srt -t medium --src-lang en

# Full robustness run (default ASR preset):
vas subtitle tests/data/nightofthelivingdead_1968_en.mp4 \
    -o /tmp/notld.en.srt -t large-v3-turbo --src-lang en

# Exercise transcription + translation (English -> Korean), keep both files:
vas subtitle tests/data/nightofthelivingdead_1968_en.mp4 \
    -o /tmp/notld.ko.srt -t large-v3-turbo -T balanced \
    --src-lang en --tgt-lang ko --keep-source
# writes /tmp/notld.ko.srt (Korean) and /tmp/notld.en.srt (English source)

# TTML (IMSC1) output instead of SRT:
vas subtitle tests/data/nightofthelivingdead_1968_en.mp4 \
    -o /tmp/notld.en.ttml -t large-v3-turbo --src-lang en

# Alternative engine — Qwen3-ASR (non-Whisper; needs the [qwen] extra + a GPU
# whose compute capability your torch build supports, or --asr-device cpu):
vas subtitle tests/data/nightofthelivingdead_1968_en.mp4 \
    -o /tmp/notld.qwen.srt -t qwen3-asr --src-lang en
```

There's no reference to compute WER against, but you can still run the
**hallucination check** — give `score_wer.py` the output with **no `--ref`**.
It skips WER and the insertion-cluster check (both need a reference) and runs the
reference-free detectors: repetition loops, consecutive duplicate cues, scattered
repeated cues, and phantom phrases. This is exactly the case real film/broadcast
audio stresses — long stretches of music, ambience, and silence are where Whisper
tends to loop or emit phantom lines.

```bash
# Transcribe, then hallucination-check (no ground truth needed):
vas subtitle tests/data/nightofthelivingdead_1968_en.mp4 \
    -o /tmp/notld.en.srt -t large-v3-turbo --src-lang en
python tests/data/score_wer.py --hyp /tmp/notld.en.srt

# Same, with the Qwen3-ASR engine:
vas subtitle tests/data/nightofthelivingdead_1968_en.mp4 \
    -o /tmp/notld.qwen.srt -t qwen3-asr --src-lang en
python tests/data/score_wer.py --hyp /tmp/notld.qwen.srt
```

Observed (this film, `-t large-v3-turbo`, 1005 cues): **no repetition loops or
insertion clusters**; the flags are all plausibly genuine dialogue — `"run" x3`
at the climax (01:31), `"i don't know" x6` scattered, and a few phantom
candidates (`oh my god`, `thank you`, `you`). ~0.41% of words land in any flagged
span. A clean run; the flags are candidates to eyeball, not confirmed errors.

Otherwise inspect the cues manually, check the run completes, and confirm the SRT
is well-formed and covers the full ~96 min.

### Ground-truth note
The Archive item ships **no human transcript** — only `*.asr.srt`, which is its
own auto-generated ASR and is very poor (`"The."`, `"A listen."`, `"The the
way."`). Useless as a reference. Public-domain films essentially never come with
human subtitle tracks, and commercial films with professional subtitles are
copyrighted — "educational use" does **not** make ripping their audio/subtitles
legal. So this file is for **robustness only**: does the pipeline run 90+ minutes
of real film audio and emit sane cues? For accuracy, use the TED-LIUM set below.

---

## tedlium_5talks_en.m4a — accuracy / WER

Five full TED talks concatenated into one ~96-minute multi-speaker file, each
paired with a **verbatim human transcript**. This is real-world *broadcast-style*
spontaneous speech (applause, accents, noise, self-corrections) — not read-aloud
audio — with a trustworthy reference, so you can compute word error rate.

- **Source**: TED-LIUM 3 long-form test split, via HuggingFace
  `distil-whisper/tedlium-long-form` (ungated). Audio is 16 kHz mono.
- **Underlying corpus**: TED-LIUM 3 (Hernandez et al., 2018).
- **License**: **CC BY-NC-ND 3.0** — free for non-commercial / educational /
  research use (which this is). Not for commercial redistribution.
- **Encoded**: AAC mono 16 kHz, ~95.9 min, ~51 MB.

Speakers / order (see `tedlium_5talks_en_manifest.tsv` for offsets):

| # | Speaker | Duration | Starts at |
|---|---|---|---|
| 1 | Aimee Mullins   | 20.8 min | 0:00 |
| 2 | Bill Gates      | 25.1 min | 20:49 |
| 3 | Daniel Kahneman | 18.3 min | 45:55 |
| 4 | James Cameron   | 16.4 min | 64:10 |
| 5 | Michael Specter | 15.4 min | 80:33 |

### Ground truth: `tedlium_5talks_en_reference.txt`
The five talk transcripts concatenated in the same order — **16,192 words**.
TED-LIUM transcripts are **normalized**: lowercase, no punctuation, apostrophes
spaced (`i 'm`, `it 's`), with occasional `<unk>` tokens. To score fairly you
**must normalize the hypothesis the same way** (lowercase, strip punctuation,
collapse whitespace) before computing WER — otherwise punctuation/casing inflate
the error rate. `tedlium_5talks_en_manifest.tsv` lists per-talk boundaries so you
can also score each talk separately.

### Scoring: WER + hallucination detection
Transcribe, then score with `score_wer.py` (in this directory):

```bash
# 1. Transcribe (real long-form run):
vas subtitle tests/data/tedlium_5talks_en.m4a \
    -o /tmp/ted.en.srt -t large-v3-turbo --src-lang en

# 2. Score against the reference (handles SRT parsing + TED-LIUM normalization):
python tests/data/score_wer.py \
    --hyp /tmp/ted.en.srt \
    --ref tests/data/tedlium_5talks_en_reference.txt
```

Same with the **Qwen3-ASR** engine (non-Whisper; needs the `[qwen]` extra + a GPU
your torch build supports, or `--asr-device cpu`) — useful for comparing WER
across engines on identical audio + reference:

```bash
vas subtitle tests/data/tedlium_5talks_en.m4a \
    -o /tmp/ted.qwen.srt -t qwen3-asr --src-lang en
python tests/data/score_wer.py \
    --hyp /tmp/ted.qwen.srt \
    --ref tests/data/tedlium_5talks_en_reference.txt
```

`score_wer.py` reports two things:

- **WER** — strips SRT/VTT markup, normalizes both sides the same way (lowercase,
  no punctuation, apostrophes rejoined so `i 'm` == `i'm`), and prints WER plus a
  substitution / deletion / insertion / hits breakdown.
- **Hallucination checks** — the failure modes Whisper-family ASR shows on
  silence, music, and long talks, each reported with its wall-clock timestamp:
  - *repetition loops* — an n-gram repeated back-to-back (`you know you know …`)
  - *duplicate cues* — the same cue text over **consecutive** cues
  - *repeated cues* — the same cue text recurring many times **scattered** through
    the file (loops/duplicates only catch adjacent repeats; Whisper sometimes
    emits the same hallucinated line over and over throughout a long transcript)
  - *phantom phrases* — standalone cues that are known idle outputs (`thank you`,
    `oh my god`, `i love you`, `thanks for watching`, `please subscribe`,
    music/applause tags, …) — flagged as candidates, since several also occur
    legitimately
  - *insertion clusters* — long runs of hypothesis words with no reference match
    (from the jiwer alignment)

It uses `jiwer` if installed (now a `dev` dependency; `.venv/bin/pip install
jiwer` if missing) and otherwise falls back to a slower built-in WER and skips
the insertion-cluster check, so it still runs with no extra deps. `--hyp` accepts
`.srt`, `.vtt`, or `.txt`. Detection thresholds are tunable —
`--loop-repeats`, `--loop-ngram`, `--dup-cues`, `--repeat-cues`,
`--repeat-min-words`, `--insertion-run`
(`python tests/data/score_wer.py --help`).

`large-v3-turbo` typically lands a few % WER on TED-LIUM; smaller presets higher.

Observed baseline (this set, `-t large-v3-turbo --src-lang en`, ~3 min on a
V100): **WER 4.92%** (308 sub / 417 del / 72 ins, 1735 cues), and **no**
repetition loops, duplicate cues, or insertion clusters. The phantom / repeated-cue
checks flag the `thank you` cues (12 phantom, 11 scattered repeats) — but the
reference itself contains `thank you` 7 times (talk closings), so those are
genuine speech, not hallucinations: a reminder that these are candidates to
eyeball, not confirmed errors. After dedup only ~0.16% of words sit in any
flagged span.

---

## Downloading / regenerating

### Night of the Living Dead
```bash
cd tests/data
curl -fL -C - --retry 3 -o nightofthelivingdead_1968_en.mp4 \
  'https://archive.org/download/Night.Of.The.Living.Dead_1080p/NightOfTheLivingDead_1080p.mp4'
```
Smaller variants (same audio) exist at the same item, e.g.
`NightOfTheLivingDead_1080p_512kb.mp4` (~388 MB).

### TED-LIUM 5-talk set
Rebuild from the HuggingFace datasets-server (no `datasets` lib needed). The five
talks are test-split rows 0, 1, 4, 7, 9 (Mullins, Gates, Kahneman, Cameron,
Specter):
```bash
cd tests/data && mkdir -p _p
URL='https://datasets-server.huggingface.co/rows?dataset=distil-whisper/tedlium-long-form&config=default&split=test&offset=0&length=11'
curl -s "$URL" -o _p/rows.json
: > _p/concat.txt ; : > tedlium_5talks_en_reference.txt
for i in 0 1 4 7 9; do
  src=$(jq -r ".rows[$i].row.audio[0].src" _p/rows.json)
  curl -fsL -o "_p/$i.wav" "$src"
  echo "file '$PWD/_p/$i.wav'" >> _p/concat.txt
  jq -r ".rows[$i].row.text" _p/rows.json | tr -s ' \n\t' ' ' >> tedlium_5talks_en_reference.txt
  printf ' ' >> tedlium_5talks_en_reference.txt
done
ffmpeg -y -f concat -safe 0 -i _p/concat.txt -c:a aac -b:a 96k -ac 1 -ar 16000 \
  -movflags +faststart tedlium_5talks_en.m4a
rm -rf _p
```
Note: the `audio[0].src` URLs are time-limited presigned links — re-fetch
`rows.json` if they expire. Pick different row indices for other talks (11 total
in the test split).

## Other properly-licensed real-world corpora

If you outgrow this set, these are real-world (not read-aloud) and cleanly
licensed — pull more hours or other languages:

- **VoxPopuli** (`facebook/voxpopuli`, **CC0**) — European Parliament broadcasts;
  transcripts + interpreted translations (good for the translate stage too).
- **CoVoST 2** — English audio + reference translations into zh / ja and more;
  lets you score transcription (WER) *and* translation (BLEU).
- **TED-LIUM 3 full** (OpenSLR #51) — the complete 452-hour corpus if you want
  more than the 11-talk test split.
