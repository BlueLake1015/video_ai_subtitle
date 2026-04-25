# Test fixtures

## test_en_30s.mp4

A 30-second clip from the *Sintel* trailer (Blender Foundation, 2010), trimmed
from `https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4`
starting at the 5-second mark.

- Resolution: 1280x720, h264, ~4.7 MB
- Audio: AAC stereo, 48 kHz, English narration
- License: Creative Commons Attribution 3.0 (Blender Foundation / durian.blender.org)

Use it for smoke-testing the pipeline:

```bash
source .venv/bin/activate
vas subtitle tests/fixtures/test_en_30s.mp4 -o /tmp/out.srt -t tiny --src-lang en
```

The `tiny` Whisper preset is fastest for a sanity check; bump to `large-v3-turbo`
for real quality.

## Regenerating

If the fixture is missing or corrupt:

```bash
cd tests/fixtures
curl -fLso sintel_trailer.mp4 https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4
ffmpeg -y -ss 5 -i sintel_trailer.mp4 -t 30 \
       -c:v libx264 -preset veryfast -crf 23 \
       -c:a aac -b:a 128k -ac 2 -movflags +faststart \
       test_en_30s.mp4
rm sintel_trailer.mp4
```
