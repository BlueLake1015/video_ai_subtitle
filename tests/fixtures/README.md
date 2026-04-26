# Test fixtures

## test_en_30s.mp4

A 30-second clip from a NASA Goddard Space Flight Center / Scientific
Visualization Studio narrated explainer:

- **Source video**: *"IPCC Projections of Temperature and Precipitation in the 21st Century"*
- **Producer**: NASA Center for Climate Simulation + NASA's Scientific Visualization Studio (SVS)
- **Date**: 27 September 2013
- **Origin URL**: https://images.nasa.gov/details/GSFC_20130927_IPCC%20m11376_21st

Why this clip:

- **Public domain** (NASA works are not subject to copyright in the U.S.)
- **English narration** explaining a scientific data visualization
- **No people on screen** — pure animated maps / graphs
- **Official video dataset** — the NASA Image and Video Library is the canonical
  public-asset distribution channel for NASA Goddard productions

Encoded properties:

- 1280×720, h264, AAC stereo 48 kHz
- 30.0 s, ~3 MB
- Re-encoded with `libx264 -crf 23 -preset veryfast` and `+faststart`

## test_zh_30s.mp4

A 30-second Mandarin-narration fixture composited from a public-domain LibriVox
audiobook reading paired with a solid-color background:

- **Source audio**: *阿Q正传* (*The True Story of Ah Q*), Chapter 1, by Lu Xun (鲁迅, 1921)
- **Reading**: LibriVox volunteer recording (public domain, modern vernacular Mandarin)
- **Origin URL**: https://archive.org/details/truestoryahq_1612_librivox
- **License**: Public Domain Mark 1.0 (the recording itself is CC0/PD; the underlying text is PD by age)

Why this clip:

- **Real human Mandarin narration** (not TTS) — the recording is by a native speaker
- **No video subject** — just a solid background; ASR tests don't need visual signal
- **Public domain** at both the recording and source-text level
- **Recognised open dataset** — LibriVox is the canonical PD audiobook archive
- **Well-enunciated literary text** — predictable, useful for ASR sanity checking

We trim out the LibriVox spoken intro (which is also in Mandarin) and start at
the first line of Lu Xun's preface: *"我要为阿Q作正传，已经不止一两年了..."*

Encoded properties:

- 1280×720, h264, AAC stereo 48 kHz, ~0.5 MB
- Audio is the real LibriVox MP3 (mono 22 kHz) re-encoded to AAC stereo 48 kHz
- Video track is a solid `#1a1a2e` color (the audio is what's tested, not the visual)

## End-to-end test commands

```bash
source .venv/bin/activate

# English -> SRT (transcription only)
vas subtitle tests/fixtures/test_en_30s.mp4 -o /tmp/en.srt \
    -t medium --src-lang en

# English -> Korean (transcription + translation, keeping both files)
vas subtitle tests/fixtures/test_en_30s.mp4 -o /tmp/en-ko.ko.srt \
    -t medium -T fast \
    --src-lang en --tgt-lang ko --keep-source
# writes /tmp/en-ko.ko.srt (Korean) and /tmp/en-ko.en.srt (English source)

# Chinese -> SRT (transcription only)
vas subtitle tests/fixtures/test_zh_30s.mp4 -o /tmp/zh.srt \
    -t medium --src-lang zh

# Chinese -> English (transcription + translation, keeping both files)
vas subtitle tests/fixtures/test_zh_30s.mp4 -o /tmp/zh-en.en.srt \
    -t medium -T fast \
    --src-lang zh --tgt-lang en --keep-source
# writes /tmp/zh-en.en.srt (English) and /tmp/zh-en.zh.srt (Chinese source)

# Chinese -> Korean (transcription + translation, keeping both files)
vas subtitle tests/fixtures/test_zh_30s.mp4 -o /tmp/zh-ko.ko.srt \
    -t medium -T fast \
    --src-lang zh --tgt-lang ko --keep-source
# writes /tmp/zh-ko.ko.srt (Korean) and /tmp/zh-ko.zh.srt (Chinese source)
```

The bundled `scripts/local_file_test.sh` runs `test_en_30s.mp4` through several
preset combinations automatically (default = `medium` + `medium`+`fast` en→ko).

## Regenerating from source

### English fixture

```bash
cd tests/fixtures
URL='https://images-assets.nasa.gov/video/GSFC_20130927_IPCC%20m11376_21st/GSFC_20130927_IPCC%20m11376_21st~medium.mp4'
curl -fLso ipcc_full.mp4 "$URL"
ffmpeg -y -ss 0 -i ipcc_full.mp4 -t 30 \
       -c:v libx264 -preset veryfast -crf 23 \
       -c:a aac -b:a 128k -ac 2 -movflags +faststart \
       test_en_30s.mp4
rm ipcc_full.mp4
```

### Chinese fixture

```bash
cd tests/fixtures
curl -fLso ahq01.mp3 https://archive.org/download/truestoryahq_1612_librivox/trueQ_01_lu_64kb.mp3
ffmpeg -y \
    -f lavfi -t 30 -i "color=c=0x1a1a2e:s=1280x720:r=30" \
    -ss 28 -t 30 -i ahq01.mp3 \
    -map 0:v -map 1:a \
    -c:v libx264 -preset veryfast -crf 23 -tune stillimage \
    -c:a aac -b:a 128k -ac 2 -ar 48000 \
    -shortest -movflags +faststart \
    test_zh_30s.mp4
rm ahq01.mp3
```

The `-ss 28` skip lands just past the spoken LibriVox intro and chapter title,
so the trimmed 30 s starts on Lu Xun's first line of prose.

## Picking different clips

- **English**: browse https://images.nasa.gov/ , filter by Media Type: Video,
  search e.g. `narrated visualization`. Each result exposes a `collection.json`
  URL with direct mp4 files at multiple resolutions, plus an `.srt`
  ground-truth caption when one exists.
- **Chinese (or any other LibriVox language)**: browse https://librivox.org/
  for the language you want; archive.org hosts every recording at
  `https://archive.org/details/<librivox_identifier>`, with multiple bitrates
  and OGG / MP3 formats.
