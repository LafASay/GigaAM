from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from .diarization import Diarizer
from .preprocess import SAMPLE_RATE, load_audio

_DIARIZERS: dict = {}

# Допустимые стратегии нарезки longform-аудио на ASR-чанки.
CHUNKING_STRATEGIES: List[str] = ["vad", "speaker", "utterance"]


def get_diarizer(device: torch.device) -> Diarizer:
    """
    Возвращает общий Diarizer для указанного устройства; модель загружается
    один раз и переиспользуется в последующих вызовах.
    """
    key = str(device)
    if key not in _DIARIZERS:
        _DIARIZERS[key] = Diarizer(device=device)
    return _DIARIZERS[key]


@dataclass
class SpeechChunks:
    """Результат нарезки аудио на ASR-чанки: волновые сегменты и их границы."""

    segments: List[Tensor]
    boundaries: List[Tuple[float, float]]
    probs: np.ndarray
    frame_shift: float


def _pack_regions(
    audio: Tensor,
    sr: int,
    regions: List[Tuple[float, float]],
    boundaries: List[Tuple[float, float]],
    max_duration: float,
    min_duration: float,
    strict_limit_duration: float,
    new_chunk_threshold: float,
) -> List[Tensor]:
    """
    Склеивает iterable из речевых интервалов в чанки жадным обходом по
    правилам max/min длительности; чанки длиннее strict_limit делятся
    равномерно. Возвращает волновые сегменты; границы дописывает в
    boundaries (список пополняется вызывающей стороной).

    ``min_duration=0.0`` отключает склейку по минимуму: каждый интервал
    длиннее ``new_chunk_threshold`` становится отдельным чанком (режим
    "utterance"), более короткие интервалы приклеиваются к следующему
    чанку. Вырожденные интервалы (за пределами аудио после обрезки) чанков
    не порождают.
    """
    segments: List[Tensor] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0

    def _flush_chunk(start: float, end: float, duration: float) -> None:
        if duration > strict_limit_duration:
            max_segments = int(duration / strict_limit_duration) + 1
            segment_duration = duration / max_segments
            seg_end = start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(start * sr) : int(seg_end * sr)])
                boundaries.append((start, seg_end))
                start = seg_end
                seg_end += segment_duration
        segments.append(audio[int(start * sr) : int(end * sr)])
        boundaries.append((start, end))

    for start, end in regions:
        start = max(0.0, start)
        end = min(audio.shape[0] / sr, end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _flush_chunk(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _flush_chunk(curr_start, curr_end, curr_duration)

    return segments


def chunk_speech(
    wav_file: str,
    sr: int = SAMPLE_RATE,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
    strategy: str = "vad",
    device: torch.device = torch.device("cpu"),
) -> SpeechChunks:
    """
    Разбивает аудиоволну на ASR-чанки по речевым областям.

    ``strategy="vad"`` (по умолчанию) — речь детектируется моделью
    диаризации (объединение активности всех спикеров, порог 0.5) вместо
    VAD pyannote; речевые области склеиваются по правилам max/min
    длительности. ``strategy="speaker"`` — аудио сначала режется по
    моментам смены доминирующего спикера, затем полученные куски
    склеиваются в чанки по тем же правилам max/min длительности.
    ``strategy="utterance"`` — каждая реплика (интервал одного
    доминирующего спикера) длиннее ``new_chunk_threshold`` становится
    отдельным чанком без склейки; более короткие интервалы (в т.ч.
    вырожденные) приклеиваются к следующему чанку.
    Чанки длиннее strict_limit делятся равномерно.
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    audio = load_audio(wav_file)
    diarizer = get_diarizer(device)
    result = diarizer.analyze(wav_file)
    frame_shift = diarizer.frame_shift
    one_chunk_per_region = strategy == "utterance"
    if strategy == "speaker" or one_chunk_per_region:
        regions = diarizer.speaker_turns(result.probs, frame_shift)
    else:
        regions = diarizer.speech_regions(result.probs, frame_shift)
    boundaries: List[Tuple[float, float]] = []

    segments = _pack_regions(
        audio,
        sr,
        regions,
        boundaries,
        max_duration=max_duration,
        # "utterance": min_duration=0.0 — каждый интервал длиннее
        # new_chunk_threshold становится отдельным чанком; интервалы
        # короче (в т.ч. вырожденные из-за паддинга диаризатора) не
        # порождают чанков, а приклеиваются к следующему.
        min_duration=0.0 if one_chunk_per_region else min_duration,
        strict_limit_duration=strict_limit_duration,
        new_chunk_threshold=new_chunk_threshold,
    )

    return SpeechChunks(
        segments=segments,
        boundaries=boundaries,
        probs=result.probs,
        frame_shift=frame_shift,
    )


def segment_audio_file(
    wav_file: str,
    sr: int = SAMPLE_RATE,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
    strategy: str = "vad",
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[Tensor], List[Tuple[float, float]]]:
    """
    Разбивает аудиоволну на более мелкие фрагменты на основе речевой активности.
    Речь детектируется моделью диаризации Nemotron-3-Diarization.

    ``strategy="vad"`` (по умолчанию) — чанки по речевым областям;
    ``strategy="speaker"`` — разрез по сменам доминирующего спикера
    с последующей склейкой кусков в чанки;
    ``strategy="utterance"`` — каждая реплика = отдельный чанк.
    """
    chunks = chunk_speech(
        wav_file,
        sr=sr,
        max_duration=max_duration,
        min_duration=min_duration,
        strict_limit_duration=strict_limit_duration,
        new_chunk_threshold=new_chunk_threshold,
        strategy=strategy,
        device=device,
    )
    return chunks.segments, chunks.boundaries
