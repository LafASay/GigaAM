# AGENTS.md — GigaAM

Семейство открытых акустических моделей (Conformer, 220M–600M) для русского/мультиязычного ASR, распознавания эмоций и SSL-эмбеддингов. Один Python-пакет `gigaam`, без монорепозиторных границ.

## Установка

- Python ≥ 3.10; **ffmpeg должен быть в PATH** (`load_audio` вызывает его через shell — отсутствие ffmpeg ломает загрузку аудио с невнятной ошибкой).
- `pip install -e ".[torch]"` — torch это *опциональный* extra (обычно уже установлен; не переустанавливайте его вслепую).
- Extra `longform` пинит `torch==2.10.*`; установка обоих extras вместе может конфликтовать по версиям torch.
- `transcribe_longform` дополнительно требует `pip install -e ".[longform]"` + переменную окружения `HF_TOKEN` с доступом к `pyannote/segmentation-3.0` на Hugging Face.

## Команды

Lint должен точно совпадать с флагами CI — **файла конфигурации** для flake8/mypy нет; флаги живут только в `.github/workflows/gigaam.yml`:

```bash
black --check --diff gigaam/ tests/          # line-length 88 (pyproject)
isort --check-only --diff gigaam/ tests/     # profile=black
flake8 --ignore=E203,W503,W504 --max-line-length=120 gigaam/ tests/
mypy gigaam/ --ignore-missing-imports --no-strict-optional   # только gigaam/, не tests
```

## Тесты

Большинство тестов скачивают реальные чекпойнты (~ГБ) с CDN Сбера в `~/.cache/gigaam` и тестовые WAV через `wget` — нужен интернет, работа занимает минуты:

- Быстро/офлайн: `pytest -v tests/test_normalize.py` (чистая нормализация текста, без скачиваний).
- Дефолт CI: `pytest -v tests/test_loading.py -m partial` — облегчённое подмножество.
- `-m full` скачивает **каждый** чекпойнт (очень медленно, много места на диске; после использования удаляет каждый ckpt).
- CI запускает каждый файл отдельно: `test_reading`, `test_batching`, `test_longform` (нужен HF_TOKEN), `test_onnx`, `test_timestamps`.
- Особенности ONNX/GPU проверяются в `test_onnx.py`; CI работает на CPU, поэтому никогда не предполагайте CUDA в тестах.

## Архитектура / ловушки

- Точка входа: `gigaam.load_model(name)` (`gigaam/__init__.py`) — принимает ревизию модели (`v3_*`, см. `_MODEL_HASHES`) **или локальный путь `.ckpt`** после файнтюна. Поддержка v1/v2/emo/multilingual удалена.
- Иерархия классов: `GigaAM` (только SSL-энкодер) → `GigaAMASR` (+ голова/декодирование). Веса проверяются контрольной суммой md5 — никогда не обходите валидацию `_MODEL_HASHES`.
- `model.transcribe()` бросает исключение для аудио > 25 с (`LONGFORM_THRESHOLD`); длинное аудио должно идти через `transcribe_longform()` (сегментация через VAD pyannote).
- `to_onnx` по умолчанию экспортирует в fp32; для GPU-инференса передавайте `dtype=torch.float16`. Экспорт приводит модуль к нужному типу, а затем возвращает `module.float()`.
- Фикс паддинга сабсэмплинг-свёртки (`StridingSubsampling._mask_time` в `gigaam/encoder.py`) сохраняет батчевый выход равным выходу с батчем размера 1 — test_batching зависит от него; не удаляйте.

## Конвенции

- Докстринги и README на русском (недавно переведены); идентификаторы/комментарии в коде на английском. Сохраняйте язык окружения: документация — русский, код — английский.
- Стиль коммитов: короткие, конвенциональные (`docs: ...`, `Fix: ...` или обычное предложение).
- `triton_scripts/` (развёртывание через Triton Inference Server / TensorRT) имеет собственный README и Docker-сетап — считайте отдельной подсистемой.
