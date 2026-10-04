import hashlib
import logging
import os
import time
import urllib.request
import warnings
from typing import Optional, Tuple, Union

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .model import GigaAM, GigaAMASR
from .preprocess import load_audio
from .utils import format_time, normalize_raw_text

__all__ = [
    "GigaAM",
    "GigaAMASR",
    "load_audio",
    "format_time",
    "load_model",
    "normalize_raw_text",
]

# Default cache directory
_CACHE_DIR = os.path.expanduser("~/.cache/gigaam")
# Url with model checkpoints
_URL_DIR = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
_DOWNLOAD_RETRIES = 3
_MODEL_HASHES = {
    "v3_ctc": "73413e7be9c6a5935827bfab5c0dd678",
    "v3_rnnt": "0fd2c9a1ff66abd8d32a3a07f7592815",
    "v3_e2e_ctc": "367074d6498f426d960b25f49531cf68",
    "v3_e2e_rnnt": "2730de7545ac43ad256485a462b0a27a",
    "v3_ssl": "70cbf5ed7303a0ed242ddb257e9dc6a6",
}


def _download_file(
    file_url: str, file_path: str, retries: int = _DOWNLOAD_RETRIES
) -> str:
    """Скачивает файл, если его ещё нет в кэше, с несколькими повторными попытками при сбое."""
    if os.path.exists(file_path):
        return file_path

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with (
                urllib.request.urlopen(file_url) as source,
                open(file_path, "wb") as output,
            ):
                with tqdm(
                    total=int(source.info().get("Content-Length", 0)),
                    ncols=80,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))
            return file_path
        except BaseException as err:
            if os.path.exists(file_path):
                os.remove(file_path)
            if attempt == retries or not isinstance(err, Exception):
                raise
            logging.warning(
                "Download of %s failed (attempt %d/%d): %s. Retrying...",
                file_url,
                attempt,
                retries,
                err,
            )
            time.sleep(2)

    raise RuntimeError(f"Download of {file_url} failed after {retries} attempts.")


def _download_model(model_name: str, download_root: str) -> Tuple[str, str]:
    """Скачивает веса модели, если они ещё не закэшированы."""
    short_names = ["ctc", "rnnt", "e2e_ctc", "e2e_rnnt", "ssl"]
    possible_names = short_names + list(_MODEL_HASHES.keys())
    if model_name not in possible_names:
        raise ValueError(
            f"Model '{model_name}' not found. Available model names: {possible_names}"
        )

    if model_name in short_names:
        model_name = f"v3_{model_name}"
    model_url = f"{_URL_DIR}/{model_name}.ckpt"
    model_path = os.path.join(download_root, model_name + ".ckpt")
    return model_name, _download_file(model_url, model_path)


def _download_tokenizer(model_name: str, download_root: str) -> Optional[str]:
    """При необходимости скачивает токенизатор и возвращает путь к нему."""
    if "e2e" not in model_name:
        return None  # No tokenizer required for this model

    tokenizer_url = f"{_URL_DIR}/{model_name}_tokenizer.model"
    tokenizer_path = os.path.join(download_root, model_name + "_tokenizer.model")
    return _download_file(tokenizer_url, tokenizer_path)


def hash_path(ckpt_path: str) -> str:
    """Вычисляет хэш бинарного файла для проверки контрольной суммы"""
    return hashlib.md5(open(ckpt_path, "rb").read()).hexdigest()


def _apply_flash_policy(cfg, use_flash: Optional[bool], device_obj: torch.device):
    """Применяет переопределение flash_attn и откат на CPU к конфигурации модели."""
    if use_flash is not None:
        cfg.encoder.flash_attn = use_flash
    if cfg.encoder.get("flash_attn", False) and device_obj.type == "cpu":
        logging.warning("flash_attn is not supported on CPU. Disabling it...")
        cfg.encoder.flash_attn = False


def _finalize_model(
    model: Union["GigaAM", "GigaAMASR"],
    fp16_encoder: bool,
    device_obj: torch.device,
) -> Union["GigaAM", "GigaAMASR"]:
    """Общий финал загрузки: режим eval, опциональный fp16-энкодер, целевое устройство."""
    model = model.eval()
    if fp16_encoder and device_obj.type != "cpu":
        model.encoder = model.encoder.half()
    return model.to(device_obj)


def _normalize_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Приводит параметр устройства к torch.device."""
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_str)
    if isinstance(device, str):
        return torch.device(device)
    return device


def load_model(
    model_name: str,
    fp16_encoder: bool = True,
    use_flash: Optional[bool] = False,
    device: Optional[Union[str, torch.device]] = None,
    download_root: Optional[str] = None,
) -> Union[GigaAM, GigaAMASR]:
    """
    Загружает модель GigaAM по имени или локальный ``.ckpt`` после файнтюна.

    Параметры
    ----------
    model_name : str
        Имя модели или путь к файлу ``.ckpt``.
    fp16_encoder:
        Нужно ли преобразовать веса энкодера к точности FP16.
    use_flash : Optional[bool]
        Использовать ли flash_attn, если модель это позволяет (требуется установленная библиотека flash_attn).
        По умолчанию False.
    device : Optional[Union[str, torch.device]]
        Устройство, на которое загрузить модель. По умолчанию "cuda", если доступно, иначе "cpu".
    download_root : Optional[str]
        Каталог для скачивания модели. По умолчанию "~/.cache/gigaam".
    """
    device_obj = _normalize_device(device)

    if download_root is None:
        download_root = _CACHE_DIR

    local_path = os.path.expanduser(model_name)
    if os.path.isfile(local_path):
        finetuned = torch.load(local_path, map_location="cpu", weights_only=False)
        hparams = finetuned["hyper_parameters"]
        base_name = hparams["model_name"]
        sd = {
            k: v
            for k, v in finetuned["state_dict"].items()
            if k.startswith(("preprocessor.", "encoder.", "head."))
        }
        model_cfg = hparams.get("model_cfg")
        if model_cfg is not None:
            cfg = OmegaConf.create(model_cfg)
            if not (cfg.get("decoding") or {}).get("model_path"):
                _apply_flash_policy(cfg, use_flash, device_obj)
                asr_model = GigaAMASR(cfg)
                asr_model.load_state_dict(sd)
                return _finalize_model(asr_model, fp16_encoder, device_obj)

        model = load_model(
            base_name,
            fp16_encoder=fp16_encoder,
            use_flash=use_flash,
            device=device_obj,
            download_root=download_root,
        )
        model.load_state_dict(sd)
        return model

    model_name, model_path = _download_model(model_name, download_root)
    tokenizer_path = _download_tokenizer(model_name, download_root)

    assert (
        hash_path(model_path) == _MODEL_HASHES[model_name]
    ), f"Model checksum failed. Please run `rm {model_path}` and reload the model"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=(FutureWarning))
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    _apply_flash_policy(checkpoint["cfg"], use_flash, device_obj)

    if tokenizer_path is not None:
        checkpoint["cfg"].decoding.model_path = tokenizer_path

    if "ssl" in model_name:
        model = GigaAM(checkpoint["cfg"])
    else:
        model = GigaAMASR(checkpoint["cfg"])

    model.load_state_dict(checkpoint["state_dict"])
    checkpoint["cfg"].model_name = model_name
    return _finalize_model(model, fp16_encoder, device_obj)
