from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from .preprocess import SAMPLE_RATE, load_audio
from .types import DiarizationSegment

_HF_MODEL_ID = "nvidia/Nemotron-3-Diarization"
_SPEECH_PROB_THRESHOLD = 0.5


class DiarizationResult(NamedTuple):
    """Полный результат диаризации: сегменты и карта вероятностей (T, num_speakers)."""

    segments: List[DiarizationSegment]
    probs: np.ndarray


class Diarizer:
    """
    Спикер-диаризация на модели NVIDIA Nemotron-3-Diarization (Sortformer).

    До 8 спикеров, каналы упорядочены по времени первого появления спикера
    в аудио. Модель не gated и не требует Hugging Face токена. Оффлайн-режим
    сам разбивает длинное аудио на чанки — ограничений по длительности нет.

    Карта вероятностей ``(num_frames, num_speakers)`` с кадром 10 мс
    переиспользуется для сегментации longform-аудио (замена VAD pyannote)
    и для атрибуции спикеров словам.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device
        self._model = None
        self._processor = None

    @property
    def device(self) -> torch.device:
        """Устройство инференса; по умолчанию — авто (cuda → mps → cpu)."""
        if self._device is None:
            from . import _normalize_device

            self._device = _normalize_device(None)
        return self._device

    @property
    def model(self) -> torch.nn.Module:
        """Лениво загружает модель и процессор из transformers."""
        if self._model is None:
            from transformers import (
                AutoModelForAudioFrameClassification,
                AutoProcessor,
            )

            self._processor = AutoProcessor.from_pretrained(_HF_MODEL_ID)
            self._model = (
                AutoModelForAudioFrameClassification.from_pretrained(
                    _HF_MODEL_ID, dtype=torch.float32
                )
                .eval()
                .to(self.device)
            )
        return self._model

    @property
    def frame_shift(self) -> float:
        """Длительность одного выходного кадра, в секундах (0.01)."""
        self.model
        assert self._processor is not None
        extractor = self._processor.feature_extractor
        return extractor.hop_length / extractor.sampling_rate

    def analyze_array(self, wav: np.ndarray) -> DiarizationResult:
        """
        Диаризует волну (np.float32, SAMPLE_RATE) и возвращает сегменты вместе
        с полной картой вероятностей активности спикеров (num_frames, 8).
        """
        model = self.model
        assert self._processor is not None
        inputs = self._processor(
            wav, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        probs = logits.sigmoid()[0].cpu().numpy()
        return DiarizationResult(
            segments=self._to_segments(probs, self.frame_shift), probs=probs
        )

    def analyze(self, wav_file: str) -> DiarizationResult:
        """
        Диаризует аудиофайл: сегменты (start, end, speaker) и карта вероятностей.
        """
        return self.analyze_array(load_audio(wav_file).numpy())

    def diarize(self, wav_file: str) -> List[DiarizationSegment]:
        """Диаризует аудиофайл и возвращает список сегментов (start, end, speaker)."""
        return self.analyze(wav_file).segments

    @staticmethod
    def _to_segments(probs: np.ndarray, frame_shift: float) -> List[DiarizationSegment]:
        """Бинаризует карту вероятностей порогом и собирает сегменты спикеров."""
        num_frames = probs.shape[0]
        raw: List[Tuple[int, float, float]] = []
        for speaker in range(probs.shape[1]):
            start: Optional[int] = None
            for i, flag in enumerate(probs[:, speaker] >= _SPEECH_PROB_THRESHOLD):
                if flag and start is None:
                    start = i
                elif not flag and start is not None:
                    raw.append((speaker, start * frame_shift, i * frame_shift))
                    start = None
            if start is not None:
                raw.append((speaker, start * frame_shift, num_frames * frame_shift))
        raw.sort(key=lambda item: item[1])
        return [
            DiarizationSegment(start=start, end=end, speaker=speaker)
            for speaker, start, end in raw
        ]

    @staticmethod
    def speech_regions(
        probs: np.ndarray, frame_shift: float
    ) -> List[Tuple[float, float]]:
        """
        Речевые области как объединение активности всех спикеров
        (замена VAD pyannote при сегментации longform-аудио).
        """
        active = np.any(probs >= _SPEECH_PROB_THRESHOLD, axis=1)
        regions: List[Tuple[float, float]] = []
        start: Optional[int] = None
        for i, flag in enumerate(active):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                regions.append((start * frame_shift, i * frame_shift))
                start = None
        if start is not None:
            regions.append((start * frame_shift, float(probs.shape[0]) * frame_shift))
        return regions

    @staticmethod
    def dominant_speaker(
        probs: np.ndarray, start: float, end: float, frame_shift: float
    ) -> Optional[int]:
        """
        Спикер с максимальной суммарной вероятностью активности на интервале
        [start, end] (секунды); None, если интервал не покрывает ни одного кадра.
        """
        frame_start = max(0, int(round(start / frame_shift)))
        frame_end = min(probs.shape[0], int(round(end / frame_shift)))
        if frame_start >= frame_end:
            return None
        return int(np.argmax(probs[frame_start:frame_end].sum(axis=0)))
