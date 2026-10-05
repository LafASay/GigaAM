# GigaAM: семейство открытых акустических моделей для обработки речи

<div align="center" style="line-height: 1;">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01192-b31b1b.svg)](https://arxiv.org/abs/2506.01192)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow.svg)](https://huggingface.co/collections/ai-sage/gigaam)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salute-developers/GigaAM/blob/main/colab_example.ipynb)

</div>

<hr>

## Последние обновления
* **2026/10** — [диаризация спикеров](#диаризация-спикеров) на модели [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization): сегментация длинных аудио и определение "кто говорил когда" (до 8 спикеров) без Hugging Face токена
* **2026/04** — [дообучение моделей](#дообучение-моделей) (CTC / RNNT), таймстемпы на уровне слов, [Triton Inference Server](#triton-inference-server-и-tensorrt)
* **2025/11** — GigaAM-v3: снижение WER на **30%** на новых доменах данных; GigaAM-v3-e2e: end-to-end распознавание речи (**70:30** в side-by-side сравнении против Whisper-large-v3)
* **2025/06** — Наша [научная статья о GigaAM](https://arxiv.org/abs/2506.01192) принята на InterSpeech 2025!
* **2024/05** — [Поддержка распознавания речи на длинных аудиозаписях](#основные-функции)

---

## Установка

### Требования
- Python ≥ 3.13
- [ffmpeg](https://ffmpeg.org/) установлен и добавлен в переменную PATH системы

### Установка пакета GigaAM

```bash
# Клонировать репозиторий
git clone https://github.com/salute-developers/GigaAM.git  
cd GigaAM

# Установить зависимости
pip install -e .[torch]

# (опционально) Проверить установку:
pip install -e ".[tests]"
pytest -v tests/test_loading.py -m partial  # или `-m full` для тестирования всех моделей
```

---

## Обзор GigaAM

GigaAM - акустическая модель на базе архитектуры [Conformer](https://arxiv.org/pdf/2005.08100.pdf) (~220M параметров), предобученная на русскоязычных речевых данных. Она служит основой для всего семейства GigaAM и обеспечивает высокое качество при дообучении на задачи распознавания речи. Для задач автоматического распознавания речи (ASR) мы дообучили энкодер GigaAM с декодерами на основе [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) и [RNNT](https://arxiv.org/abs/1211.3711). Доступна одна линейка моделей:

| | Метод предобучения | Объём предобучения (часы) | Объём данных ASR (часы) | Доступные версии |
| :--- | :--- | :--- | :--- | :---: |
| **v3** | HuBERT–CTC | 700 000 | 4 000 | `v3_ssl`, `v3_ctc`, `v3_rnnt`, `v3_e2e_ctc`, `v3_e2e_rnnt` |

Версии `v3_e2e_ctc` и `v3_e2e_rnnt` поддерживают пунктуацию и нормализацию текста.

## Качество моделей

В обучение `GigaAM-v3` были включены новые внутренние наборы данных: колл-центр, музыка, речь с атипичными характеристиками и голосовые сообщения. В результате модели в среднем демонстрируют улучшение на **30%** (по метрике WER) на новых доменах данных. В сравнении end-to-end моделей (`e2e_ctc` и `e2e_rnnt`) с Whisper (оценка проводилась с использованием внешней LLM в формате side-by-side) модели GigaAM выигрывают в соотношении **70:30**.

---

## Использование
### Основные функции

**Важно:** функция `.transcribe` для ASR применима только к аудиофайлам **до 25 секунд**. Для длинных аудио и диаризации установите дополнительные зависимости:

```bash
pip install -e ".[longform]"
# опционально: запустить тесты длинной транскрибации и диаризации
pip install -e ".[tests]"
pytest -v tests/test_longform.py tests/test_diarization.py
```

Нарезка длинного аудио и диаризация выполняются открытой моделью [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) — она не gated и не требует Hugging Face токена (веса (~380 МБ) скачиваются в кэш Hugging Face при первом запуске).

<br>


```python
import gigaam

# Путь к вашему аудиофайлу
audio_path = "audio.wav"

# Аудио-эмбеддинги
model_name = "v3_ssl"       # Единственный вариант ssl-энкодера
model = gigaam.load_model(model_name)
embedding, _ = model.embed_audio(audio_path)
print(embedding)

# Распознавание речи
model_name = "v3_e2e_rnnt"  # Варианты: любые версии с суффиксами `_ctc` или `_rnnt`
model = gigaam.load_model(model_name)
transcription = model.transcribe(audio_path)
print(transcription)

# Распознавание речи с таймстемпами на уровне слов
result = model.transcribe(audio_path, word_timestamps=True)
for word in result.words:
    print(f"  [{word.start:.2f} - {word.end:.2f}] {word.text}")

# Распознавание на длинном аудио
result = model.transcribe_longform(long_audio_path)
for segment in result:
   print(f"[{gigaam.format_time(segment.start)} - {gigaam.format_time(segment.end)}]: {segment.text}")
```

### Диаризация спикеров

Модель [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (Sortformer, до 8 спикеров) определяет "кто говорил когда". Индекс спикера соответствует порядку его первого появления в аудио.

```python
# Диаризация без транскрибации: список сегментов (start, end, speaker)
for seg in model.diarize(long_audio_path):
    print(f"[{seg.start:.2f} - {seg.end:.2f}] speaker_{seg.speaker}")

# Транскрибация длинного аудио со спикерами на сегментах и словах
result = model.transcribe_longform(long_audio_path, word_timestamps=True, diarize=True)
for segment in result:
    print(f"[{gigaam.format_time(segment.start)} - {gigaam.format_time(segment.end)}] "
          f"speaker_{segment.speaker}: {segment.text}")
    for word in segment.words:
        print(f"    [{word.start:.2f} - {word.end:.2f}] speaker_{word.speaker}: {word.text}")
```

Одна и та же модель диаризации используется и для нарезки длинного аудио на ASR-чанки (объединение активности всех спикеров заменяет VAD), поэтому дополнительных зависимостей не требуется.

### REST API (FastAPI)

`server.py` в корне репозитория — простая FastAPI-обёртка: эндпоинт `POST /asr` принимает аудиофайл и возвращает JSON или markdown со временем, спикером и текстом каждой реплики (подряд идущие слова одного спикера склеиваются). Длительность аудио не ограничена — под капотом используется `transcribe_longform` с диаризацией.

```bash
pip install -e ".[longform,api]"

# запуск из корня репозитория
uvicorn server:app --host 0.0.0.0 --port 8000
```

Примеры запросов:

```bash
# JSON (по умолчанию)
curl -F "file=@audio.wav" http://localhost:8000/asr

# markdown-таблица (Время | Спикер | Реплика)
curl -F "file=@audio.wav" "http://localhost:8000/asr?format=markdown"

# markdown-диалог
curl -F "file=@audio.wav" "http://localhost:8000/asr?format=markdown&style=dialog"
```

Пример JSON-ответа:

```json
{
  "transcription": "Вечерня отошла давно, Но в кельях тихо и темно...",
  "utterances": [
    {
      "start": 0.0,
      "end": 19.0,
      "start_h": "00:00:00:00",
      "end_h": "00:00:19:00",
      "speaker": 0,
      "text": "Вечерня отошла давно, Но в кельях тихо и темно..."
    }
  ]
}
```

Модель и устройство настраиваются переменными окружения: `GIGAAM_MODEL` (по умолчанию `v3_e2e_rnnt`), `GIGAAM_DEVICE` (`cuda`/`mps`/`cpu`, по умолчанию авто) и `GIGAAM_FP16` (по умолчанию включён). Проверка работоспособности — `GET /health`.

### Загрузка из Hugging Face

> Используйте установку зависимостей из [примера](./colab_example.ipynb).

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("ai-sage/GigaAM-v3", revision="e2e_rnnt", trust_remote_code=True)
```

### Конвертация в ONNX и использование графа

> **Примечание:** `to_onnx` по умолчанию экспортирует в **fp32**. Для GPU рекомендуется передать `dtype=torch.float16` — это ускоряет инференс и снижает потребление VRAM. GPU будет использоваться после удаления onnxruntime и установки `pip install onnxruntime-gpu==1.22.*`.

1. Экспорт модели в ONNX с помощью метода `model.to_onnx`:
   ```python
   onnx_dir = "onnx"
   model_version = "v3_ctc"  # Варианты: любая версия модели

   model = gigaam.load_model(model_version)
   model.to_onnx(dir_path=onnx_dir, dtype=torch.float32)  # или fp16 (рекомендовано для GPU)
   ```

2. Запуск с использованием ONNX:
   ```python
   from gigaam.onnx_utils import load_onnx, infer_onnx

   sessions, model_cfg = load_onnx(onnx_dir, model_version)
   result = infer_onnx([audio_path], model_cfg, sessions)
    print(result[0])  # str для ctc / rnnt версий, np.ndarray для ssl

   # или для целого датасета
   texts = infer_onnx("/path/to/eval/manifest.tsv", model_cfg, sessions)
   print(texts[0])
   ```

Эти и более продвинутые примеры (кастомная загрузка аудио, батчинг) доступны в [Colab notebook](https://colab.research.google.com/github/salute-developers/GigaAM/blob/main/colab_example.ipynb).

### Triton Inference Server и TensorRT

Все модели распознавания речи также можно использовать в серверном окружении в формате ONNX/TRT через Triton Inference Server. Инструкции по настройке, конвертации моделей и развёртыванию описаны в [документации Triton Inference Server](./triton_scripts/README.md).

---

## Citation

Если применяете GigaAM в своих исследованиях, используйте ссылку на нашу статью:

```bibtex
@inproceedings{kutsakov25_interspeech,
  title     = {{GigaAM: Efficient Self-Supervised Learner for Speech Recognition}},
  author    = {Aleksandr Kutsakov and Alexandr Maximenko and Georgii Gospodinov and Pavel Bogomolov and Fyodor Minkin},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {1213--1217},
  doi       = {10.21437/Interspeech.2025-1616},
  issn      = {2958-1796},
}
```

## Ссылки
* [[arxiv] GigaAM: Efficient Self-Supervised Learner for Speech Recognition](https://arxiv.org/abs/2506.01192)
* [[habr] GigaAM-v3: открытая SOTA-модель распознавания речи на русском](https://habr.com/ru/companies/sberdevices/articles/973160/)
* [[habr] GigaAM: класс открытых моделей для обработки звучащей речи](https://habr.com/ru/companies/sberdevices/articles/805569)
* [[youtube] Как научить LLM слышать: GigaAM 🤝 GigaChat Audio](https://www.youtube.com/watch?v=O7NSH2SAwRc)
* [[youtube] GigaAM: Семейство акустических моделей для русского языка](https://youtu.be/PvZuTUnZa2Q?t=26442)
* [[youtube] Speech-only Pre-training: обучение универсального аудиоэнкодера](https://www.youtube.com/watch?v=ktO4Mx6UMNk)
