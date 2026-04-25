from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    """A frame of 16 kHz mono s16le PCM samples, decoded from ffmpeg stdout.

    pcm is a float32 ndarray normalized to [-1, 1] (what Silero / Whisper expect).
    pts_s is seconds from the start of the audio stream, derived from sample count.
    """
    pcm: np.ndarray
    pts_s: float
    sample_rate: int = 16000

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / self.sample_rate


@dataclass(frozen=True)
class SpeechRegion:
    """A contiguous speech segment identified by VAD."""
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Word:
    """One transcribed word with timing, in the original stream's time base."""
    text: str
    start_s: float
    end_s: float
    probability: float = 1.0


@dataclass
class Cue:
    """A subtitle cue ready for writing.

    `lines` is the rendered display text split to respect max_chars_per_line.
    `words` preserves the per-word timing that produced the cue, useful for
    karaoke-style output and re-layout.
    """
    index: int
    start_s: float
    end_s: float
    lines: list[str]
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


CueKind = Literal["partial", "final"]


@dataclass
class StreamEvent:
    """Event emitted to WebSocket clients."""
    kind: CueKind
    start_s: float
    end_s: float
    text: str
    words: list[Word] = field(default_factory=list)
    stream_id: str | None = None

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "start": self.start_s,
            "end": self.end_s,
            "text": self.text,
            "words": [
                {"text": w.text, "start": w.start_s, "end": w.end_s, "p": w.probability}
                for w in self.words
            ],
            "stream_id": self.stream_id,
        }
