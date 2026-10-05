from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from .diarization import Diarizer
from .preprocess import SAMPLE_RATE, load_audio

_DIARIZERS: dict = {}


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


def chunk_speech(
    wav_file: str,
    sr: int = SAMPLE_RATE,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> SpeechChunks:
    """
    Разбивает аудиоволну на ASR-чанки по речевым областям.

    Речь детектируется моделью диаризации (объединение активности всех
    спикеров, порог 0.5) вместо VAD pyannote. Речевые области склеиваются
    в чанки по правилам max/min длительности; чанки длиннее
    ``strict_limit_duration`` делятся равномерно.
    """
    audio = load_audio(wav_file)
    diarizer = get_diarizer(device)
    result = diarizer.analyze(wav_file)
    frame_shift = diarizer.frame_shift
    regions = diarizer.speech_regions(result.probs, frame_shift)

    segments: List[Tensor] = []
    boundaries: List[Tuple[float, float]] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0

    def _update_segments(curr_start: float, curr_end: float, curr_duration: float):
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
        boundaries.append((curr_start, curr_end))

    audio_duration = audio.shape[0] / sr
    for start, end in regions:
        start = max(0.0, start)
        end = min(audio_duration, end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

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
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[Tensor], List[Tuple[float, float]]]:
    """
    Разбивает аудиоволну на более мелкие фрагменты на основе речевой активности.
    Речь детектируется моделью диаризации Nemotron-3-Diarization.
    """
    chunks = chunk_speech(
        wav_file,
        sr=sr,
        max_duration=max_duration,
        min_duration=min_duration,
        strict_limit_duration=strict_limit_duration,
        new_chunk_threshold=new_chunk_threshold,
        device=device,
    )
    return chunks.segments, chunks.boundaries
