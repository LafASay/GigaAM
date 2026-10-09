import logging
import os
import tempfile
from typing import List, Tuple

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

import gigaam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_predictions = {
    "v3_e2e_rnnt": [
        {
            "transcription": "Вечерня отошла давно, Но в кельях тихо и темно; Уже и сам игумен строгий Свои молитвы прекратил И кости ветхие склонил, Перекрестясь на одр убогий. Кругом и сон, и тишина; Но церкви дверь отворена; Трепещет луч лампады",  # noqa: E501
            "boundaries": (0.0, 19.0),
        },
        {
            "transcription": "И тускло озаряет он и тёмную живопись икон, и возглащённые оклады. И раздаётся в тишине то тяжкий вздох, то шёпот важный, И мрачно дремлет в тишине старинный свод, Глухой и влажный. Стоят за клиросом чернец и грешник.",  # noqa: E501
            "boundaries": (19.49, 37.54),
        },
        {
            "transcription": "Неподвижны оба, И шёпот их, как глас из гроба, И грешник бледен, как мертвец — монах. Несчастный! Полно, перестань! Ужасна исповедь злодея, Заплачена тобою дань Тому, Кто в злобе пламенея",  # noqa: E501
            "boundaries": (37.89, 55.86),
        },
        {
            "transcription": "Лукаво грешника блюдёт И к вечной гибели ведёт. Смирись, опомнись, Время, время, раскаянье, покров — Я разрешу тебя, грехов сложи мучительное бремя.",  # noqa: E501
            "boundaries": (56.1, 70.93),
        },
    ],
}


def generate_long_audio(duration=60.0, sr=16000, include_silence=True):
    """Генерирует длинное тестовое аудио с речеподобными сегментами и тишиной"""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t, dtype=np.float32)
    segment_durations = list(np.random.uniform(0.2, 5, size=100))
    current_time = 0.0

    for i, seg_duration in enumerate(segment_durations):
        if current_time + seg_duration > duration:
            break
        seg_t = np.linspace(0, seg_duration, int(sr * seg_duration))
        freq1, freq2, freq3 = 100 + i * 20, 200 + i * 30, 300 + i * 40
        segment = (
            0.4 * np.sin(2 * np.pi * freq1 * seg_t)
            + 0.3 * np.sin(2 * np.pi * freq2 * seg_t)
            + 0.2 * np.sin(2 * np.pi * freq3 * seg_t)
            + 0.1 * np.random.normal(0, 0.2, len(seg_t))
        )
        envelope = signal.windows.tukey(len(segment), alpha=0.1)
        segment = segment * envelope
        start_idx, end_idx = (
            int(current_time * sr),
            int(current_time * sr) + len(segment),
        )
        audio[start_idx:end_idx] = segment
        if include_silence and i < len(segment_durations) - 1:
            current_time += seg_duration + np.random.uniform(0.1, 0.5)
        else:
            current_time += seg_duration
    return audio


def validate_segmentation_boundaries(
    boundaries: List[Tuple[float, float]], audio_duration: float
):
    """Проверяет, что границы сегментации соответствуют требованиям"""
    issues = []
    total_duration = 0.0

    for i, (start, end) in enumerate(boundaries):
        duration = end - start
        if duration < 0.2:
            issues.append(f"Segment {i} too short: {duration:.2f}s")
        if duration > 30.0:
            issues.append(f"Segment {i} too long: {duration:.2f}s")
        if start >= end:
            issues.append(f"Segment {i} invalid boundaries: {start:.2f}-{end:.2f}")
        total_duration += duration

    if boundaries and boundaries[-1][1] > audio_duration:
        issues.append(
            f"Last segment exceeds audio: {boundaries[-1][1]:.2f} > {audio_duration:.2f}"
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_segments": len(boundaries),
    }


@pytest.mark.parametrize("strategy", ["vad", "speaker", "utterance"])
def test_segmentation_strategy(long_audio, strategy):
    """Все стратегии на long_example.wav дают валидные границы и чанки."""
    from gigaam.segmentation import chunk_speech

    chunks = chunk_speech(long_audio, strategy=strategy)

    assert chunks.boundaries, "Should produce chunk boundaries"
    duration = max(end for _, end in chunks.boundaries)
    validation = validate_segmentation_boundaries(chunks.boundaries, duration + 1.0)
    assert validation["valid"], f"Boundary validation failed: {validation['issues']}"
    assert len(chunks.segments) == len(chunks.boundaries)
    logger.info("strategy=%s: %d chunks", strategy, len(chunks.boundaries))


def test_utterance_chunks_match_speaker_turns(long_audio):
    """utterance-стратегия: границы чанков = интервалам доминирующего спикера."""
    from gigaam.diarization import Diarizer
    from gigaam.segmentation import chunk_speech

    chunks_vad = chunk_speech(long_audio, strategy="vad")
    chunks_ut = chunk_speech(long_audio, strategy="utterance")

    turns = Diarizer.speaker_turns(chunks_ut.probs, chunks_ut.frame_shift)
    assert len(chunks_ut.boundaries) == len(turns)
    assert chunks_ut.boundaries == turns
    # utterance не склеивает: чанков не меньше, чем у speaker-стратегии
    assert len(chunks_ut.boundaries) >= len(chunks_vad.boundaries)


def test_utterance_pack_no_degenerate_chunks():
    """utterance (min_duration=0.0): микро-блипы и регионы за концом аудио
    не порождают пустых/сверхкоротких чанков (регрессия ZeroDivision)."""
    import torch

    from gigaam.segmentation import _pack_regions

    sr = 16000
    audio = torch.zeros(sr * 30)
    regions = [
        (0.0, 10.0),  # обычная реплика
        (10.0, 10.01),  # микро-блит диаризации (0.01 c)
        (10.5, 20.0),  # обычная реплика
        (30.5, 31.5),  # регион в паддинге — за концом аудио
        (29.0, 29.5),  # короткая реплика перед вырожденным регионом
    ]
    boundaries: List[Tuple[float, float]] = []
    segments = _pack_regions(
        audio,
        sr,
        regions,
        boundaries,
        max_duration=22.0,
        min_duration=0.0,
        strict_limit_duration=30.0,
        new_chunk_threshold=0.2,
    )

    assert len(segments) == len(boundaries)
    for (start, end), seg in zip(boundaries, segments):
        assert end - start > 0.2, f"too short chunk: {(start, end)}"
        assert len(seg) > 0, f"empty chunk: {(start, end)}"
        assert end <= 30.0, f"chunk beyond audio: {(start, end)}"
    # блит (10.0, 10.01) приклеился к следующей реплике
    assert (10.0, 20.0) in boundaries


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_transcribe_longform_utterance(revision, long_audio):
    """utterance + word_timestamps не падает на вырожденных чанках."""
    from gigaam.types import LongformTranscriptionResult

    model = gigaam.load_model(revision)
    result = model.transcribe_longform(
        long_audio, strategy="utterance", word_timestamps=True
    )

    assert isinstance(result, LongformTranscriptionResult)
    assert len(result.segments) > 0
    for seg in result.segments:
        assert seg.text.strip(), f"Empty text for chunk {(seg.start, seg.end)}"
        assert seg.words, "word_timestamps=True should produce words"


def test_speaker_turns_synthetic():
    """speaker_turns режет по сменам доминирующего спикера, тишина не покрывается."""
    from gigaam.diarization import Diarizer

    probs = np.array(
        [
            [0.9, 0.1],  # spk0
            [0.8, 0.2],  # spk0
            [0.2, 0.8],  # spk1
            [0.1, 0.9],  # spk1
            [0.1, 0.1],  # тишина
            [0.9, 0.1],  # spk0
            [0.9, 0.1],  # spk0
        ]
    )
    turns = Diarizer.speaker_turns(probs, frame_shift=0.01)
    assert turns == [(0.0, 0.02), (0.02, 0.04), (0.05, 0.07)]


def test_merged_speaker_chunks_duration_limits(duration=90.0):
    """speaker-стратегия: чанки укладываются в strict_limit после склейки кусков."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = generate_long_audio(duration=duration)
            sf.write(f.name, audio, 16000)

            from gigaam.segmentation import chunk_speech

            chunks = chunk_speech(f.name, strategy="speaker")

            assert chunks.boundaries, "Speaker strategy should produce chunks"
            for start, end in chunks.boundaries:
                assert end - start <= 30.0, f"Chunk too long: {end - start:.2f}s"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


@pytest.mark.parametrize("duration", [30.0, 60.0, 120.0])
def test_segmentation_functionality(duration):
    """Проверяет сегментацию аудио с разной длительностью"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = generate_long_audio(duration=duration)
            sf.write(f.name, audio, 16000)

            from gigaam.segmentation import segment_audio_file

            segments, boundaries = segment_audio_file(f.name, sr=16000)

            validation = validate_segmentation_boundaries(boundaries, duration)
            assert validation[
                "valid"
            ], f"Boundary validation failed: {validation['issues']}"
            assert len(segments) == len(
                boundaries
            ), "Segments and boundaries count mismatch"

            logger.info(f"Segmentation: {len(segments)} segments for {duration}s audio")

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_transcribe_longform(revision, long_audio):
    """Проверяет longform-транскрипцию для разных моделей"""
    from gigaam.types import LongformTranscriptionResult, Segment

    model = gigaam.load_model(revision)
    result = model.transcribe_longform(long_audio)
    ref = _predictions[revision]

    assert isinstance(
        result, LongformTranscriptionResult
    ), "Should return LongformTranscriptionResult"
    assert len(result.segments) == len(ref), "Distinct results len from reference"

    for segment, ref_segment in zip(result.segments, ref):
        assert isinstance(segment, Segment), "Should be Segment object"
        assert hasattr(segment, "text"), "Missing text attribute"
        assert hasattr(segment, "start"), "Missing start attribute"
        assert hasattr(segment, "end"), "Missing end attribute"
        start, end = segment.start, segment.end
        ref_start, ref_end = ref_segment["boundaries"]
        assert (
            abs(start - ref_start) < 0.1 and abs(end - ref_end) < 0.1
        ), f"Segments are not close {start, end} and {ref_start, ref_end}"
        assert (
            segment.text == ref_segment["transcription"]
        ), f"Different transcription: {segment.text} and {ref_segment['transcription']}"


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_longform_consistency(revision):
    """Проверяет, что повторные запуски дают согласованные результаты"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = generate_long_audio(duration=30.0)
            sf.write(f.name, audio, 16000)

            model = gigaam.load_model(revision)
            result1 = model.transcribe_longform(f.name)
            result2 = model.transcribe_longform(f.name)

            assert len(result1.segments) == len(
                result2.segments
            ), "Inconsistent segment count"
            for seg1, seg2 in zip(result1.segments, result2.segments):
                assert (seg1.start, seg1.end) == (
                    seg2.start,
                    seg2.end,
                ), "Inconsistent boundaries"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_segmentation_edge_cases():
    """Проверяет сегментацию на крайних случаях"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            # Very short audio
            audio = generate_long_audio(duration=0.5)
            sf.write(f.name, audio, 16000)

            from gigaam.segmentation import segment_audio_file

            segments, boundaries = segment_audio_file(f.name, sr=16000)

            # Should handle short audio gracefully
            assert isinstance(segments, list), "Should return list even for short audio"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
