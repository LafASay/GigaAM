from typing import List, Optional, Tuple

import hydra
import omegaconf
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .preprocess import SAMPLE_RATE, load_audio
from .types import (
    DiarizationSegment,
    LongformTranscriptionResult,
    Segment,
    TranscriptionResult,
    Word,
)
from .utils import AudioDataset, onnx_converter

LONGFORM_THRESHOLD = 25 * SAMPLE_RATE


class GigaAM(nn.Module):
    """
    Giga Acoustic Model: самостоятельная (self-supervised) модель для речевых задач
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)

    def forward(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Выполняет прямой проход через препроцессор и энкодер.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        if self._device.type == "cpu":
            return self.encoder(features, feature_lengths)
        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            return self.encoder(features, feature_lengths)

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_wav(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Подготавливает аудиофайл к обработке: загружает его
        на нужное устройство и конвертирует формат.
        """
        wav = load_audio(wav_file)
        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)
        return wav, length

    def embed_audio(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Извлекает представления аудио с помощью модели GigaAM.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Экспортирует ONNX-энкодер модели в указанный каталог.
        """
        with self.encoder.onnx_export_mode():
            self._to_onnx(dir_path, dtype=dtype)
        omegaconf.OmegaConf.save(self.cfg, f"{dir_path}/{self.cfg.model_name}.yaml")

    def diarize(self, wav_file: str) -> List["DiarizationSegment"]:
        """
        Определяет "кто говорил когда" в аудиофайле с помощью модели
        Nemotron-3-Diarization. Длительность аудио не ограничена.
        """
        from .diarization import Diarizer

        return Diarizer(device=self._device).diarize(wav_file)

    def _to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Экспортирует ONNX-энкодер модели в указанный каталог.
        """
        onnx_converter(
            model_name=f"{self.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.encoder,
            dynamic_axes=self.encoder.dynamic_axes(),
            export_dtype=dtype,
        )


class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model для распознавания речи
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    def _decode(
        self,
        encoded: Tensor,
        encoded_len: Tensor,
        wav_lens: Tensor,
        word_timestamps: bool = False,
    ) -> List[Tuple[str, Optional[List[Word]]]]:
        decoded = self.decoding.decode(self.head, encoded, encoded_len)
        if not word_timestamps:
            return [(t, None) for t, _, _ in decoded]
        from .timestamps_utils import compute_frame_shift, frames_to_words

        out: List[Tuple[str, Optional[List[Word]]]] = []
        for i, (text, token_ids, token_frames) in enumerate(decoded):
            frame_shift = compute_frame_shift(
                int(wav_lens[i].item()), int(encoded_len[i].item())
            )
            out.append(
                (
                    text,
                    frames_to_words(
                        self.decoding.tokenizer,
                        token_ids,
                        token_frames,
                        frame_shift,
                    ),
                )
            )
        return out

    @torch.inference_mode()
    def transcribe(
        self, wav_file: str, word_timestamps: bool = False
    ) -> TranscriptionResult:
        """
        Транскрибирует короткий аудиофайл в текст.
        Возвращает TranscriptionResult с опциональными таймстампами на уровне слов.
        """
        wav, length = self.prepare_wav(wav_file)
        if length.item() > LONGFORM_THRESHOLD:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

        encoded, encoded_len = self.forward(wav, length)
        text, words = self._decode(encoded, encoded_len, length, word_timestamps)[0]
        return TranscriptionResult(text=text, words=words)

    def forward_for_export(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Прямой проход энкодер-декодер для сохранения модели целиком в формате onnx.
        """
        encoded, encoded_len = self.encoder(features, feature_lengths)
        return self.head(encoded), encoded_len

    def _to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Экспортирует ONNX ASR-модель.
        `ctc`:  экспортируется целиком в формате энкодер-декодер.
        `rnnt`: экспортируется по частям: энкодер/декодер/joint.
        """
        if "ctc" in self.cfg.model_name:
            saved_forward = self.forward
            self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
            try:
                onnx_converter(
                    model_name=self.cfg.model_name,
                    out_dir=dir_path,
                    module=self,
                    inputs=self.encoder.input_example(),
                    input_names=["features", "feature_lengths"],
                    output_names=["log_probs", "encoded_lengths"],
                    dynamic_axes={
                        "features": {0: "batch_size", 2: "seq_len"},
                        "feature_lengths": {0: "batch_size"},
                        "log_probs": {0: "batch_size", 1: "seq_len"},
                        "encoded_lengths": {0: "batch_size"},
                    },
                    export_dtype=dtype,
                )
            finally:
                self.forward = saved_forward  # type: ignore[assignment, method-assign]
        else:
            super()._to_onnx(dir_path, dtype=dtype)
            onnx_converter(
                model_name=f"{self.cfg.model_name}_decoder",
                out_dir=dir_path,
                module=self.head.decoder,
                dynamic_axes=self.head.decoder.dynamic_axes(),
                export_dtype=dtype,
            )
            onnx_converter(
                model_name=f"{self.cfg.model_name}_joint",
                out_dir=dir_path,
                module=self.head.joint,
                dynamic_axes=self.head.joint.dynamic_axes(),
                export_dtype=dtype,
            )

    @torch.inference_mode()
    def transcribe_longform(
        self,
        wav_file: str,
        word_timestamps: bool = False,
        diarize: bool = False,
        fr_batch_size: int = 16,
        fr_num_workers: int = 0,
        **kwargs,
    ) -> LongformTranscriptionResult:
        """
        Транскрибирует длинный аудиофайл, разбивая его на сегменты и
        затем транскрибируя каждый сегмент (батчевый инференс через AudioDataset).
        Управляйте батчевым инференсом через fr_batch_size и fr_num_workers.
        При diarize=True сегментам и словам присваивается индекс спикера
        (модель Nemotron-3-Diarization; она же определяет речевые области
        для нарезки).
        Возвращает LongformTranscriptionResult с сегментами, содержащими
        опциональные таймстампы на уровне слов и опциональных спикеров.
        """
        from .diarization import Diarizer
        from .segmentation import chunk_speech

        chunks = chunk_speech(wav_file, SAMPLE_RATE, device=self._device, **kwargs)
        segments, boundaries = chunks.segments, chunks.boundaries

        if not segments:
            return LongformTranscriptionResult(segments=[])

        ds = AudioDataset(segments, tokenizer=None)
        dl = DataLoader(
            ds,
            batch_size=fr_batch_size,
            shuffle=False,
            collate_fn=AudioDataset.collate,
            num_workers=fr_num_workers,
        )

        result_segments: List[Segment] = []
        idx = 0
        for wav_pad, wav_lens in dl:
            wav_pad = wav_pad.to(self._device).to(self._dtype)
            wav_lens = wav_lens.to(self._device)
            encoded, encoded_len = self.forward(wav_pad, wav_lens)
            for text, words in self._decode(
                encoded, encoded_len, wav_lens, word_timestamps
            ):
                seg_start, seg_end = boundaries[idx]
                idx += 1
                seg_speaker = None
                if diarize:
                    seg_speaker = Diarizer.dominant_speaker(
                        chunks.probs, seg_start, seg_end, chunks.frame_shift
                    )
                if word_timestamps:
                    out_words: List[Word] = []
                    for w in words or []:
                        w_speaker = None
                        if diarize:
                            w_speaker = Diarizer.dominant_speaker(
                                chunks.probs,
                                seg_start + w.start,
                                seg_start + w.end,
                                chunks.frame_shift,
                            )
                        out_words.append(
                            Word(
                                text=w.text,
                                start=round(w.start + seg_start, 3),
                                end=round(w.end + seg_start, 3),
                                speaker=w_speaker,
                            )
                        )
                    result_segments.append(
                        Segment(
                            text=text,
                            start=seg_start,
                            end=seg_end,
                            words=out_words,
                            speaker=seg_speaker,
                        )
                    )
                else:
                    result_segments.append(
                        Segment(
                            text=text,
                            start=seg_start,
                            end=seg_end,
                            speaker=seg_speaker,
                        )
                    )
        return LongformTranscriptionResult(segments=result_segments)
