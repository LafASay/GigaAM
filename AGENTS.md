# AGENTS.md — GigaAM

Открытая акустическая модель (Conformer, ~220M параметров) для русского ASR. Доступна модель `v3_e2e_rnnt`; для нарезки длинных аудио и диаризации используется модель [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization). Один Python-пакет `gigaam`, без монорепозиторных границ.

## Установка

- Python 3.13; **ffmpeg должен быть в PATH** (`load_audio` вызывает его через shell — отсутствие ffmpeg ломает загрузку аудио с невнятной ошибкой).
- `pip install -e .` — все зависимости (включая torch, transformers, тестовые и линтер-пакеты) в одном списке без extras.
- `transcribe_longform` и диаризация используют transformers ≥ 5.18. Модель диаризации [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) не gated — HF_TOKEN не нужен; веса (~380 МБ) кэшируются HF-хабом при первом запуске.

## Команды

Lint должен точно совпадать с флагами CI — **файла конфигурации** для flake8/mypy нет; флаги живут только в `.github/workflows/gigaam.yml` (примечание: YAML воркфлоу сейчас не парсится — копируйте флаги дословно). Версии линтеров: black==26.1.0, isort==7.0.0, flake8==7.3.0:

```bash
black --check --diff gigaam/ tests/          # line-length 88 (pyproject)
isort --check-only --diff gigaam/ tests/     # profile=black
flake8 --ignore=E203,W503,W504 --max-line-length=120 gigaam/ tests/
mypy gigaam/ --ignore-missing-imports --no-strict-optional   # только gigaam/, не tests
```

## Тесты

Тестовые WAV закоммичены в `tests/assets/` (`example.wav`, `long_example.wav`) — сеть для аудио не нужна. Чекпойнты моделей (~ГБ) скачиваются с CDN Сбера в `gigaam_cache/` в корне репозитория при первом обращении `load_model` (переопределяется `GIGAAM_CACHE_DIR`; fallback `~/.cache/gigaam`, если корень репозитория не записываем; каталог в .gitignore — используется для офлайн-сборки контейнера, см. `scripts/preload_models.py`). Для longform/диаризации нужен снапшот `nvidia/Nemotron-3-Diarization` (не gated, токен не нужен): он качается через HF-hub, но `Diarizer` передаёт `cache_dir` явно — по умолчанию `gigaam_cache/hf` (переопределяется `HF_HUB_CACHE` или параметром `Diarizer(cache_dir=...)`). Работа занимает минуты:

- Быстро/офлайн: `pytest -v tests/test_normalize.py` (чистая нормализация текста).
- Дефолт CI: `pytest -v tests/test_loading.py -m partial` — облегчённое подмножество.
- `-m full` повторно скачивает чекпойнт `v3_e2e_rnnt` (медленно; после использования удаляет каждый ckpt).
- Общие аудио-фикстуры (`test_audio`, `long_audio`) живут в `tests/conftest.py`.
- CI запускает каждый файл отдельно: `test_reading`, `test_batching`, `test_longform`, `test_diarization`, `test_onnx`, `test_timestamps`.
- Особенности ONNX/GPU проверяются в `test_onnx.py`; CI работает на CPU, поэтому никогда не предполагайте CUDA в тестах.
- Тесты сверяют точные строки транскрипций/нормализации (`_predictions` в `test_loading.py`, `test_normalize.py`) — любые правки нормализации/декодирования не должны ломать эти эталоны.

## Архитектура / ловушки

- Точка входа: `gigaam.load_model(name)` (`gigaam/__init__.py`) — принимает имя `v3_e2e_rnnt` (см. `_MODEL_HASHES`; алиасы удалены) **или локальный путь `.ckpt`** после файнтюна. Поддержка v1/v2/emo/multilingual и прочих ревизий (ssl/ctc/rnnt/e2e_ctc) удалена.
- Иерархия классов: `GigaAM` (базовый класс: препроцессор + энкодер) → `GigaAMASR` (+ RNNT-голова/декодирование). Веса проверяются контрольной суммой md5 — никогда не обходите валидацию `_MODEL_HASHES`.
- `model.transcribe()` бросает исключение для аудио > 25 с (`LONGFORM_THRESHOLD`); длинное аудио должно идти через `transcribe_longform()` (нарезка и диаризация через Nemotron-3-Diarization, `gigaam/segmentation.py` + `gigaam/diarization.py`).
- Официально код рассчитан на CPU/CUDA, но MPS тоже поддержан: `_normalize_device` (`gigaam/__init__.py`) автовыбирает `cuda → mps → cpu`, а `test_batching` использует `device_type=model._device.type`. На Apple Silicon отмечено: fp16-энкодер, транскрипция/word_timestamps/longform совпадают с эталоном.
- `to_onnx` по умолчанию экспортирует в fp32; для GPU-инференса передавайте `dtype=torch.float16`. Экспорт приводит модуль к нужному типу, а затем возвращает `module.float()`.
- Фикс паддинга сабсэмплинг-свёртки (`StridingSubsampling._mask_time` в `gigaam/encoder.py`) сохраняет батчевый выход равным выходу с батчем размера 1 — test_batching зависит от него; не удаляйте.

## Конвенции

- Докстринги и README на русском (недавно переведены); идентификаторы/комментарии в коде на английском. Сохраняйте язык окружения: документация — русский, код — английский.
- Стиль коммитов: короткие, конвенциональные (`docs: ...`, `Fix: ...` или обычное предложение).
- `triton_scripts/` (развёртывание через Triton Inference Server / TensorRT) имеет собственный README и Docker-сетап — считайте отдельной подсистемой.
