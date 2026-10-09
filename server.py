"""
Простой FastAPI-сервис поверх GigaAM: транскрибация + диаризация спикеров.

Эндпоинты
---------
POST /asr  — аудио (multipart) -> JSON или markdown со временем, спикером и репликой
GET /health — состояние сервиса

Запуск
------
    pip install -e .
    uvicorn server:app --host 0.0.0.0 --port 8000

Переменные окружения
--------------------
GIGAAM_MODEL  — ревизия модели ASR (по умолчанию v3_e2e_rnnt)
GIGAAM_DEVICE — "cuda", "mps" или "cpu" (по умолчанию авто-выбор cuda -> mps -> cpu)
GIGAAM_FP16   — fp16-энкодер на GPU/MPS (по умолчанию включён)"""

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Iterator, List, Literal, Optional, Tuple

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, computed_field

import gigaam
from gigaam.types import LongformTranscriptionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gigaam.server")

_DEFAULT_MODEL = "v3_e2e_rnnt"
_ASR_MODEL: Optional[gigaam.GigaAMASR] = None


class Utterance(BaseModel):
    """Реплика: подряд идущие слова одного спикера, склеенные вместе."""

    start: float
    end: float
    speaker: Optional[int] = None
    text: str

    @computed_field
    @property
    def start_h(self) -> str:
        """Начало реплики в формате HH:MM:SS:mm."""
        return gigaam.format_time(self.start)

    @computed_field
    @property
    def end_h(self) -> str:
        """Конец реплики в формате HH:MM:SS:mm."""
        return gigaam.format_time(self.end)


class AsrResponse(BaseModel):
    """JSON-ответ POST /asr."""

    transcription: str
    utterances: List[Utterance]


@asynccontextmanager
async def lifespan(_: FastAPI) -> Iterator[None]:
    """Загружает ASR-модель один раз при старте приложения."""
    global _ASR_MODEL
    model_name = os.getenv("GIGAAM_MODEL", _DEFAULT_MODEL)
    device = os.getenv("GIGAAM_DEVICE") or None
    fp16 = os.getenv("GIGAAM_FP16", "true").lower() not in ("0", "false", "no")
    logger.info(
        "Loading ASR model %s (device=%s, fp16=%s)...", model_name, device, fp16
    )
    model = gigaam.load_model(model_name, fp16_encoder=fp16, device=device)
    if not isinstance(model, gigaam.GigaAMASR):
        raise RuntimeError(
            f"Model '{model_name}' does not support transcription; "
            "set GIGAAM_MODEL to v3_e2e_rnnt"
        )
    _ASR_MODEL = model
    logger.info("Model %s ready on %s", model.cfg.model_name, model._device)
    yield
    _ASR_MODEL = None


app = FastAPI(title="GigaAM ASR", lifespan=lifespan)


def _require_model() -> gigaam.GigaAMASR:
    """Возвращает загруженную модель или 503, если она ещё не готова."""
    if _ASR_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return _ASR_MODEL


def _merge_utterances(result: LongformTranscriptionResult) -> List[Utterance]:
    """
    Склеивает подряд идущие слова одного спикера в реплики.

    Спикер сегмента — доминирующий на всём ASR-чанке (до ~22 с), поэтому
    склейка по сегментам теряла смену спикера внутри чанка; слова же
    атрибутируются индивидуально. Сегменты без слов (запрошена транскрипция
    без таймстампов) учитываются целиком как реплика спикера сегмента.
    """
    spans: List[Tuple[str, float, float, Optional[int]]] = []
    for seg in result.segments:
        if seg.words:
            spans.extend((w.text, w.start, w.end, w.speaker) for w in seg.words)
        elif seg.text.strip():
            spans.append((seg.text.strip(), seg.start, seg.end, seg.speaker))

    utterances: List[Utterance] = []
    for text, start, end, speaker in spans:
        text = text.strip()
        if not text:
            continue
        last = utterances[-1] if utterances else None
        if last is not None and last.speaker == speaker:
            last.end = end
            last.text = f"{last.text} {text}"
        else:
            utterances.append(
                Utterance(start=start, end=end, speaker=speaker, text=text)
            )
    return utterances


def _md_cell(text: str) -> str:
    """Экранирует текст для ячейки markdown-таблицы."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _render_markdown(utterances: List[Utterance], style: str) -> str:
    """Рендерит реплики в markdown: таблица или диалоговые блоки."""
    if style == "dialog":
        return "\n\n".join(
            f"**Спикер {u.speaker}** [{u.start_h}]: {u.text}" for u in utterances
        )
    rows = ["| Время | Спикер | Реплика |", "| --- | --- | --- |"]
    for u in utterances:
        rows.append(
            f"| {u.start_h} - {u.end_h} | Спикер {u.speaker} | {_md_cell(u.text)} |"
        )
    return "\n".join(rows)


def _transcribe_uploaded(
    model: gigaam.GigaAMASR, file: UploadFile, strategy: str
) -> LongformTranscriptionResult:
    """Сохраняет загруженный файл во временный и транскрибирует его со спикерами."""
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        return model.transcribe_longform(
            tmp_path, word_timestamps=True, diarize=True, strategy=strategy
        )
    finally:
        os.unlink(tmp_path)


@app.post("/asr", response_model=AsrResponse)
def asr(
    file: UploadFile = File(
        ..., description="Аудиофайл в любом формате, поддерживаемом ffmpeg"
    ),
    format: Literal["json", "markdown"] = Query(
        "json", description="Формат ответа: json или markdown"
    ),
    style: Literal["table", "dialog"] = Query(
        "table", description="Стиль markdown: таблица или диалоговые блоки"
    ),
    chunking: Literal["vad", "speaker", "utterance"] = Query(
        "vad",
        description=(
            "Стратегия нарезки длинного аудио на ASR-чанки: vad — речевые "
            "области (по умолчанию), speaker — разрез по смене доминирующего "
            "спикера со склейкой, utterance — одна реплика = один чанк"
        ),
    ),
):
    """
    Транскрибирует аудио и определяет спикеров (модель Nemotron-3-Diarization).

    Подряд идущие слова одного спикера склеиваются в реплики. По умолчанию
    возвращается JSON; `?format=markdown` отдаёт таблицу или диалог
    (`&style=dialog`) со временем, спикером и текстом каждой реплики.
    Стратегия нарезки длинного аудио задаётся `?chunking=...`
    (vad / speaker / utterance).
    Длительность аудио не ограничена (используется transcribe_longform).
    """
    model = _require_model()
    file.file.seek(0, os.SEEK_END)
    if file.file.tell() == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    file.file.seek(0)

    try:
        result = _transcribe_uploaded(model, file, chunking)
    except HTTPException:
        raise
    except RuntimeError as exc:
        # ffmpeg inside load_audio could not decode the input
        raise HTTPException(
            status_code=400, detail=f"Failed to read audio: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("ASR failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    utterances = _merge_utterances(result)
    logger.info("asr: %d utterances from %s", len(utterances), file.filename)
    if format == "markdown":
        return PlainTextResponse(
            _render_markdown(utterances, style), media_type="text/markdown"
        )
    return AsrResponse(
        transcription=" ".join(u.text for u in utterances), utterances=utterances
    )


@app.get("/health")
def health():
    """Состояние сервиса: имя модели и устройство инференса."""
    if _ASR_MODEL is None:
        return {"status": "loading", "model": None, "device": None}
    return {
        "status": "ok",
        "model": _ASR_MODEL.cfg.model_name,
        "device": _ASR_MODEL._device.type,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
