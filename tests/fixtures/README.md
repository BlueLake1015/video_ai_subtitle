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

Encoded properties of the trimmed fixture:

- 1280x720, h264, AAC stereo 48 kHz
- 30.0 s, ~3 MB
- Re-encoded with `libx264 -crf 23 -preset veryfast` and `+faststart` so the
  decoder finds keyframes immediately at any seek

## End-to-end test command

```bash
source .venv/bin/activate

# Transcription only
vas subtitle tests/fixtures/test_en_30s.mp4 -o /tmp/out.srt \
    -t medium --src-lang en

# Transcription + translation (English -> Korean), keeping both files
vas subtitle tests/fixtures/test_en_30s.mp4 -o /tmp/out.ko.srt \
    -t medium -T fast \
    --src-lang en --tgt-lang ko --keep-source
# writes /tmp/out.ko.srt (Korean) and /tmp/out.en.srt (English source)
```

The bundled `scripts/local_file_test.sh` runs this fixture through several
preset combinations automatically (default = `medium` + `medium`+`fast` en→ko).

## Regenerating from source

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

## Picking a different official clip

Browse https://images.nasa.gov/ , filter by `Media Type: Video`, search e.g.
`narrated visualization` or `animation explained`. Each result page exposes a
`collection.json` URL listing direct mp4 files at multiple resolutions, plus an
`.srt` ground-truth caption when one exists.
