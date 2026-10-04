import warnings
from subprocess import CalledProcessError, run
from typing import Tuple

import torch
import torchaudio
from torch import Tensor, nn

SAMPLE_RATE = 16000


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> Tensor:
    """
    Загружает аудиофайл и передискретизирует его до заданной частоты дискретизации.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        audio = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError("Failed to load audio") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return torch.frombuffer(audio, dtype=torch.int16).float() / 32768.0


class SpecScaler(nn.Module):
    """
    Модуль, применяющий логарифмическое масштабирование к значениям спектрограммы.
    Сначала ограничивает входные значения заданным диапазоном, затем применяет натуральный логарифм.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(nn.Module):
    """
    Модуль извлечения признаков Log-mel спектрограммы из сырых аудиосигналов.
    Использует преобразование MelSpectrogram из Torchaudio для извлечения признаков
    и применяет логарифмическое масштабирование.
    """

    def __init__(self, sample_rate: int, features: int, **kwargs):
        super().__init__()
        self.hop_length = kwargs.get("hop_length", sample_rate // 100)
        self.win_length = kwargs.get("win_length", sample_rate // 40)
        self.n_fft = kwargs.get("n_fft", sample_rate // 40)
        self.center = kwargs.get("center", True)
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=features,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                center=self.center,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Вычисляет длину выхода после процесса извлечения признаков.
        """
        if self.center:
            return (
                input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()
            )
        else:
            return (
                (input_lengths - self.win_length)
                .div(self.hop_length, rounding_mode="floor")
                .add(1)
                .long()
            )

    def forward(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Извлекает признаки Log-mel спектрограммы из входного аудиосигнала.
        """
        return self.featurizer(input_signal), self.out_len(length)
