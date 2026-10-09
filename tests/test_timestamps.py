import logging

import pytest

import gigaam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_predictions = {
    "v3_e2e_rnnt": {
        "text": "Ничьих не требуя похвал, Счастлив уж я надеждой сладкой, Что дева с трепетом любви Посмотрит, может быть, украдкой На песни грешные мои. У лукоморья дуб зелёный.",  # noqa: E501
        "words": [
            {"word": "Ничьих", "start": 0.04, "end": 0.4},
            {"word": "не", "start": 0.52, "end": 0.56},
            {"word": "требуя", "start": 0.64, "end": 0.96},
            {"word": "похвал,", "start": 1.08, "end": 1.6},
            {"word": "Счастлив", "start": 1.72, "end": 2.16},
            {"word": "уж", "start": 2.24, "end": 2.4},
            {"word": "я", "start": 2.48, "end": 2.52},
            {"word": "надеждой", "start": 2.64, "end": 3.12},
            {"word": "сладкой,", "start": 3.16, "end": 3.68},
            {"word": "Что", "start": 3.72, "end": 3.76},
            {"word": "дева", "start": 3.88, "end": 4.08},
            {"word": "с", "start": 4.16, "end": 4.2},
            {"word": "трепетом", "start": 4.24, "end": 4.72},
            {"word": "любви", "start": 4.8, "end": 5.04},
            {"word": "Посмотрит,", "start": 5.32, "end": 6.0},
            {"word": "может", "start": 6.08, "end": 6.12},
            {"word": "быть,", "start": 6.28, "end": 6.48},
            {"word": "украдкой", "start": 6.52, "end": 6.96},
            {"word": "На", "start": 7.16, "end": 7.2},
            {"word": "песни", "start": 7.28, "end": 7.56},
            {"word": "грешные", "start": 7.68, "end": 8.08},
            {"word": "мои.", "start": 8.24, "end": 8.72},
            {"word": "У", "start": 9.2, "end": 9.24},
            {"word": "лукоморья", "start": 9.36, "end": 10.0},
            {"word": "дуб", "start": 10.12, "end": 10.36},
            {"word": "зелёный.", "start": 10.48, "end": 11.08},
        ],
    },
}


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_word_timestamps_predictions(revision, test_audio):
    """Проверяет, что таймстампы слов совпадают с ожидаемыми значениями."""
    model = gigaam.load_model(revision, device="cpu")
    result = model.transcribe(test_audio, word_timestamps=True)
    expected = _predictions[revision]

    assert result.text == expected["text"], f"Text mismatch: {result.text}"
    assert len(result.words) == len(expected["words"]), "Word count mismatch"

    for actual, exp in zip(result.words, expected["words"]):
        assert actual.text == exp["word"], f"Word mismatch: {actual} vs {exp}"
        assert abs(actual.start - exp["start"]) < 0.1, f"Start mismatch: {actual}"
        assert abs(actual.end - exp["end"]) < 0.1, f"End mismatch: {actual}"

    logger.info(f"{revision}: Word timestamps predictions matched")


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_transcribe_word_timestamps_structure(revision, test_audio):
    """Проверяет, что word_timestamps=True возвращает корректную структуру."""
    from gigaam.types import TranscriptionResult, Word

    model = gigaam.load_model(revision)
    result = model.transcribe(test_audio, word_timestamps=True)

    assert isinstance(result, TranscriptionResult), "Should return TranscriptionResult"
    assert hasattr(result, "text"), "Result should have 'text' attribute"
    assert hasattr(result, "words"), "Result should have 'words' attribute"
    assert isinstance(result.text, str), "'text' should be string"
    assert isinstance(result.words, list), "'words' should be list"
    assert all(
        isinstance(w, Word) for w in result.words
    ), "All words should be Word objects"
    logger.info(f"{revision}: text={result.text[:50]}...")


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_transcribe_word_timestamps_values(revision, test_audio):
    """Проверяет, что таймстампы слов валидны и упорядочены."""
    model = gigaam.load_model(revision)
    result = model.transcribe(test_audio, word_timestamps=True)

    words = result.words
    assert len(words) > 0, "Should have at least one word"

    prev_end = 0.0
    for w in words:
        assert hasattr(w, "text"), "Word entry should have 'text' attribute"
        assert hasattr(w, "start"), "Word entry should have 'start' attribute"
        assert hasattr(w, "end"), "Word entry should have 'end' attribute"
        assert isinstance(w.text, str), "'text' should be string"
        assert w.start < w.end, f"start should be < end: {w}"
        assert w.start >= prev_end - 0.01, f"Words should be ordered: {w}"
        prev_end = w.end

    logger.info(f"{revision}: {len(words)} words, last end={prev_end:.2f}s")


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_transcribe_default_returns_string(revision, test_audio):
    """Проверяет, что поведение по умолчанию (word_timestamps=False) возвращает TranscriptionResult с __str__."""
    from gigaam.types import TranscriptionResult

    model = gigaam.load_model(revision)
    result = model.transcribe(test_audio)

    assert isinstance(result, TranscriptionResult), "Should return TranscriptionResult"
    assert isinstance(str(result), str), "str(result) should return string"
    assert len(str(result)) > 0, "Transcription should not be empty"
    assert result.words is None, "Should not have words when word_timestamps=False"


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_transcribe_longform_word_timestamps(revision, long_audio):
    """Проверяет longform-транскрипцию с word_timestamps=True."""
    from gigaam.types import LongformTranscriptionResult, Segment, Word

    model = gigaam.load_model(revision)
    result = model.transcribe_longform(long_audio, word_timestamps=True)

    assert isinstance(
        result, LongformTranscriptionResult
    ), "Should return LongformTranscriptionResult"
    assert len(result.segments) > 0, "Should have at least one segment"

    # Check segments have words
    for seg in result.segments:
        assert isinstance(seg, Segment), "Should be Segment object"
        assert (
            seg.words is not None
        ), "Segment should have words when word_timestamps=True"
        assert all(isinstance(w, Word) for w in seg.words), "All should be Word objects"

    # Check flattened words
    all_words = result.words
    assert len(all_words) > 0, "Should have at least one word"
    prev_end = 0.0
    for w in all_words:
        assert w.start < w.end, f"start should be < end: {w}"
        prev_end = w.end

    logger.info(
        f"{revision} longform: {len(all_words)} words, last end={prev_end:.2f}s"
    )


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_transcribe_longform_default(revision, long_audio):
    """Проверяет, что longform по умолчанию возвращает сегменты с транскрипцией."""
    from gigaam.types import LongformTranscriptionResult, Segment

    model = gigaam.load_model(revision)
    result = model.transcribe_longform(long_audio)

    assert isinstance(
        result, LongformTranscriptionResult
    ), "Should return LongformTranscriptionResult"
    assert len(result.segments) > 0, "Should have segments"

    for seg in result.segments:
        assert isinstance(seg, Segment), "Should be Segment object"
        assert hasattr(seg, "text"), "Segment should have 'text' attribute"
        assert hasattr(seg, "start"), "Segment should have 'start' attribute"
        assert hasattr(seg, "end"), "Segment should have 'end' attribute"
        assert (
            seg.words is None
        ), "Segment should not have words when word_timestamps=False"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
