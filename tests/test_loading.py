import logging
import os

import pytest

import gigaam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_predictions = {
    "asr": "ничьих не требуя похвал счастлив уж я надеждой сладкой что дева с трепетом любви посмотрит может быть украдкой на песни грешные мои у лукоморья дуб зеленый",  # noqa: E501
    "v3_e2e_ctc": "Ничьих, не требуя похвал, счастлив уж я надеждой сладкой, Что дева с трепетом любви посмотрит, может быть украдкой На песни грешные мои. У лукоморья дуб зелёный.",  # noqa: E501
    "v3_e2e_rnnt": "Ничьих не требуя похвал, Счастлив уж я надеждой сладкой, Что дева с трепетом любви Посмотрит, может быть, украдкой На песни грешные мои. У лукоморья дуб зелёный.",  # noqa: E501
}


def run_model_method(model, revision, test_audio):
    if "ssl" in revision:
        result = model.embed_audio(test_audio)
        assert result is not None, "SSL embedding failed"
        logger.info(f"{revision}: SSL embedding completed")

    else:
        result = model.transcribe(test_audio)
        if "e2e" in revision:
            assert _predictions[revision] == str(
                result
            ), f"Transcription failed ({revision}): {str(result)}"
        else:
            assert _predictions["asr"] == str(
                result
            ), f"Transcription failed ({revision}): {str(result)}"
        logger.info(f"{revision}: Transcription completed")


@pytest.mark.parametrize(
    "revision",
    [
        "v3_ctc",
        "v3_rnnt",
        "v3_e2e_ctc",
        "v3_e2e_rnnt",
        "v3_ssl",
    ],
)
@pytest.mark.full
def test_model_revision_full(revision, test_audio):
    """Проверяет, что конкретная версия модели загружается и обрабатывает аудио (только полный набор моделей)"""
    model = gigaam.load_model(revision)
    run_model_method(model, revision, test_audio)
    os.remove(os.path.join(gigaam._CACHE_DIR, f"{revision}.ckpt"))


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
@pytest.mark.partial
def test_model_revision_partial(revision, test_audio):
    """Проверяет, что конкретная версия модели загружается и обрабатывает аудио (частичный набор моделей)"""
    model = gigaam.load_model(revision)
    run_model_method(model, revision, test_audio)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "partial"])
