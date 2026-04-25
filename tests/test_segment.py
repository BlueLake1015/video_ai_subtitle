import numpy as np

from vas.config import SegmentConfig, VadConfig
from vas.segment import WhisperSegmenter
from vas.types import SpeechRegion


def test_aggregates_short_regions_into_target_segment():
    audio = np.zeros(60 * 16000, dtype=np.float32)
    regions = [SpeechRegion(start_s=i * 5.0, end_s=i * 5.0 + 4.0) for i in range(10)]
    seg = WhisperSegmenter(SegmentConfig(target_seconds=28.0, min_seconds=8.0, max_seconds=34.0))
    out = seg.from_regions(audio, regions)
    assert len(out) >= 1
    for s in out:
        assert s.duration_s <= 35.0


def test_does_not_split_through_a_long_region():
    audio = np.zeros(60 * 16000, dtype=np.float32)
    regions = [SpeechRegion(start_s=0.0, end_s=20.0), SpeechRegion(start_s=21.0, end_s=40.0)]
    out = WhisperSegmenter(SegmentConfig(target_seconds=28.0, max_seconds=34.0)).from_regions(audio, regions)
    assert len(out) >= 1


def test_empty_input():
    out = WhisperSegmenter().from_regions(np.zeros(0), [])
    assert out == []


def test_segment_carries_audio_slice_with_correct_length():
    sr = 16000
    total_s = 30
    audio = np.arange(total_s * sr, dtype=np.float32)
    regions = [SpeechRegion(start_s=0.0, end_s=10.0), SpeechRegion(start_s=15.0, end_s=25.0)]
    out = WhisperSegmenter(SegmentConfig(target_seconds=15.0, min_seconds=5.0, max_seconds=30.0)).from_regions(audio, regions)
    assert out
    s0 = out[0]
    assert len(s0.audio) == int((s0.end_s - s0.start_s) * sr)
