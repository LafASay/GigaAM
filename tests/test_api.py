"""Тесты FastAPI-обёртки (server.py): эндпоинты /asr и /health."""

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(scope="session")
def client():
    """Тестовый клиент приложения; lifespan загружает модель один раз."""
    from server import app

    with TestClient(app) as test_client:
        yield test_client


def _upload(test_audio):
    with open(test_audio, "rb") as f:
        return {"file": ("example.wav", f.read(), "audio/wav")}


def test_health(client):
    """GET /health сообщает модель, устройство и статус ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"]
    assert data["device"] in ("cpu", "cuda", "mps")


def test_asr_json(client, test_audio):
    """POST /asr возвращает JSON с репликами (время, спикер, текст, слова)."""
    resp = client.post("/asr", files=_upload(test_audio))
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["transcription"].strip()
    assert len(data["utterances"]) > 0
    for u in data["utterances"]:
        assert u["start"] < u["end"], u
        assert u["start_h"] and u["end_h"], u
        assert isinstance(u["speaker"], int), u
        assert u["text"].strip(), u
        # слова не возвращаются: реплика — только время, спикер и текст
        assert set(u) == {"start", "end", "start_h", "end_h", "speaker", "text"}, u
    logger.info("asr json: %r", data["transcription"][:80])


def test_asr_markdown_table(client, test_audio):
    """format=markdown возвращает таблицу Время/Спикер/Реплика."""
    resp = client.post("/asr?format=markdown", files=_upload(test_audio))
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "| Время | Спикер | Реплика |" in resp.text
    assert "Спикер 0" in resp.text
    logger.info("asr markdown table:\n%s", resp.text)


def test_asr_markdown_dialog(client, test_audio):
    """format=markdown&style=dialog возвращает диалоговые блоки."""
    resp = client.post("/asr?format=markdown&style=dialog", files=_upload(test_audio))
    assert resp.status_code == 200, resp.text
    assert "**Спикер 0**" in resp.text
    logger.info("asr markdown dialog:\n%s", resp.text)


def test_asr_empty_file(client):
    """Пустой файл отклоняется с 400."""
    resp = client.post("/asr", files={"file": ("empty.wav", b"", "audio/wav")})
    assert resp.status_code == 400


def test_asr_bad_audio(client):
    """Некорректное аудио (ffmpeg не может декодировать) даёт 400."""
    resp = client.post(
        "/asr", files={"file": ("bad.wav", b"not audio at all", "audio/wav")}
    )
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
