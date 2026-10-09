import logging

import numpy as np
import pytest

import gigaam
from gigaam.diarization import Diarizer
from gigaam.types import DiarizationSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_SPEAKERS = 8


def test_diarization_helpers_pure_numpy():
    """Проверяет speech_regions и dominant_speaker на синтетических вероятностях."""
    # 100 frames (1 s), frame 0.01s: speaker 0 active 0.0-0.4, speaker 1 active 0.4-1.0
    probs = np.zeros((100, NUM_SPEAKERS), dtype=np.float32)
    probs[:40, 0] = 0.9
    probs[40:, 1] = 0.9

    regions = Diarizer.speech_regions(probs, frame_shift=0.01)
    assert regions == [(0.0, 1.0)], f"Unexpected speech regions: {regions}"

    assert Diarizer.dominant_speaker(probs, 0.0, 0.4, 0.01) == 0
    assert Diarizer.dominant_speaker(probs, 0.4, 1.0, 0.01) == 1
    # интервал, пересекающий смену спикера: побеждает спикер с большей активностью
    assert Diarizer.dominant_speaker(probs, 0.3, 0.45, 0.01) == 0
    assert Diarizer.dominant_speaker(probs, 0.0, 0.0, 0.01) is None

    # silence in the middle splits regions
    probs2 = np.zeros((100, NUM_SPEAKERS), dtype=np.float32)
    probs2[:30, 0] = 0.9
    probs2[50:, 0] = 0.9
    regions2 = Diarizer.speech_regions(probs2, frame_shift=0.01)
    assert regions2 == [(0.0, 0.3), (0.5, 1.0)], f"Unexpected regions: {regions2}"

    # _to_segments builds per-speaker intervals sorted by start
    segments = Diarizer._to_segments(probs, frame_shift=0.01)
    assert segments == [
        DiarizationSegment(start=0.0, end=0.4, speaker=0),
        DiarizationSegment(start=0.4, end=1.0, speaker=1),
    ]


def test_diarize_structure(test_audio):
    """Проверяет структуру результата model.diarize()."""
    model = gigaam.load_model("v3_e2e_rnnt", device="cpu")
    segments = model.diarize(test_audio)

    assert isinstance(segments, list), "Should return a list"
    assert len(segments) > 0, "Should detect at least one speech segment"
    for seg in segments:
        assert isinstance(seg, DiarizationSegment), "Should be DiarizationSegment"
        assert seg.start < seg.end, f"start should be < end: {seg}"
        assert isinstance(seg.speaker, int), "speaker should be int"
        assert 0 <= seg.speaker < NUM_SPEAKERS, f"speaker out of range: {seg}"

    starts = [s.start for s in segments]
    assert starts == sorted(starts), "Segments should be sorted by start"
    speakers = {s.speaker for s in segments}
    logger.info(f"diarize: {len(segments)} segments, speakers={speakers}")


def test_diarize_single_speaker(long_audio):
    """Длинный тестовый файл — чтение одним диктором: должен быть один спикер."""
    model = gigaam.load_model("v3_e2e_rnnt", device="cpu")
    segments = model.diarize(long_audio)

    assert len(segments) > 0, "Should detect speech"
    speakers = {s.speaker for s in segments}
    assert speakers == {0}, f"Single-speaker audio, expected {{0}}, got {speakers}"
    logger.info(f"long diarize: {len(segments)} segments, total speaker 0")


def test_transcribe_longform_with_diarization(long_audio):
    """Проверяет, что diarize=True проставляет спикеров сегментам и словам."""
    from gigaam.types import LongformTranscriptionResult

    model = gigaam.load_model("v3_e2e_rnnt", device="cpu")
    result = model.transcribe_longform(long_audio, word_timestamps=True, diarize=True)

    assert isinstance(
        result, LongformTranscriptionResult
    ), "Should return LongformTranscriptionResult"
    assert len(result.segments) > 0, "Should have segments"

    for seg in result.segments:
        assert isinstance(seg.speaker, int), f"Segment should have speaker: {seg}"
        assert 0 <= seg.speaker < NUM_SPEAKERS, f"Speaker out of range: {seg}"
        assert seg.words is not None, "Should have words with word_timestamps=True"
        for w in seg.words:
            assert isinstance(w.speaker, int), f"Word should have speaker: {w}"
            assert 0 <= w.speaker < NUM_SPEAKERS, f"Word speaker out of range: {w}"

    seg_speakers = {s.speaker for s in result.segments}
    logger.info(
        f"longform+diarize: {len(result.segments)} segments, speakers={seg_speakers}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
