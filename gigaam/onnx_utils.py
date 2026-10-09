import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import omegaconf
import onnxruntime as rt
import torch
from tqdm.auto import tqdm

from .decoding import Tokenizer
from .preprocess import FeatureExtractor
from .utils import AudioDataset

warnings.simplefilter("ignore", category=UserWarning)

MAX_LETTERS_PER_FRAME = 3


def _session_float_dtype(session: rt.InferenceSession) -> np.dtype:
    """Определяет numpy-тип float по первому float-входу ONNX-сессии."""
    _type_map: Dict[str, np.dtype] = {
        "tensor(float16)": np.dtype(np.float16),
        "tensor(float)": np.dtype(np.float32),
        "tensor(double)": np.dtype(np.float64),
    }
    for inp in session.get_inputs():
        if inp.type in _type_map:
            return _type_map[inp.type]
    return np.dtype(np.float32)


def _build_inputs(session: rt.InferenceSession, values: List[np.ndarray]) -> dict:
    return {node.name: data for node, data in zip(session.get_inputs(), values)}


def _cat_states(
    states: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    hs = [s[0] for s in states]
    cs = [s[1] for s in states]
    return np.concatenate(hs, axis=1), np.concatenate(cs, axis=1)


def _split_state(
    state: Tuple[np.ndarray, np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    h, c = state
    b = h.shape[1]
    return [(h[:, i : i + 1], c[:, i : i + 1]) for i in range(b)]


def _decode_rnnt_batch(
    enc_features: np.ndarray,
    enc_len: np.ndarray,
    model_cfg: omegaconf.DictConfig,
    sessions: List[Optional[rt.InferenceSession]],
    tokenizer: Tokenizer,
) -> List[str]:
    pred_sess, joint_sess = sessions[1:]
    dtype = _session_float_dtype(pred_sess)

    enc_features = np.asarray(enc_features, dtype=dtype, order="C")
    blank_idx = len(tokenizer)
    pred_hidden = model_cfg.head.decoder.pred_hidden
    pred_rnn_layers = model_cfg.head.decoder.pred_rnn_layers
    B, _, T = enc_features.shape

    hyps: List[List[int]] = [[] for _ in range(B)]
    last_label: List[Optional[np.ndarray]] = [None] * B
    dec_state: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * B

    def emit_batch(batch_idx: List[int], t: int, fresh: bool) -> List[int]:
        idx = np.asarray(batch_idx, dtype=np.int64)
        f = enc_features[idx, :, t : t + 1]

        if fresh:
            labels = np.full((len(batch_idx), 1), blank_idx, dtype=np.int64)
            h = np.zeros((pred_rnn_layers, len(batch_idx), pred_hidden), dtype=dtype)
            c = np.zeros((pred_rnn_layers, len(batch_idx), pred_hidden), dtype=dtype)
        else:
            labels = np.concatenate([last_label[i] for i in batch_idx], axis=0)
            h, c = _cat_states([dec_state[i] for i in batch_idx])

        pred_outputs = pred_sess.run(
            [node.name for node in pred_sess.get_outputs()],
            _build_inputs(pred_sess, [labels, h, c]),
        )

        joint_outputs = joint_sess.run(
            [node.name for node in joint_sess.get_outputs()],
            _build_inputs(
                joint_sess,
                [f, pred_outputs[0].swapaxes(1, 2)],
            ),
        )

        k = joint_outputs[0][:, 0, 0, :].argmax(axis=-1)
        emit_pos = np.nonzero(k != blank_idx)[0]
        if emit_pos.size == 0:
            return []

        hidden_parts = _split_state((pred_outputs[1], pred_outputs[2]))
        out = []

        for p in emit_pos.tolist():
            bi = batch_idx[p]
            tok = int(k[p])

            hyps[bi].append(tok)
            last_label[bi] = np.array([[tok]], dtype=np.int64)
            dec_state[bi] = hidden_parts[p]
            out.append(bi)

        return out

    enc_len = np.asarray(enc_len, dtype=np.int64).reshape(-1)
    for t in range(T):
        active = np.nonzero(t < enc_len)[0].tolist()
        if not active:
            break

        for _ in range(MAX_LETTERS_PER_FRAME):
            if not active:
                break

            fresh = [i for i in active if dec_state[i] is None]
            stateful = [i for i in active if dec_state[i] is not None]

            next_active = []
            if fresh:
                next_active.extend(emit_batch(fresh, t, fresh=True))
            if stateful:
                next_active.extend(emit_batch(stateful, t, fresh=False))

            if not next_active:
                break

            active = next_active

    return [tokenizer.decode(h) for h in hyps]


def infer_onnx(
    data: Union[str, Sequence[Union[str, np.ndarray, torch.Tensor]]],
    model_cfg: omegaconf.DictConfig,
    sessions: List[rt.InferenceSession],
    preprocessor: Optional[FeatureExtractor] = None,
    tokenizer: Optional[Tokenizer] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    progress: bool = True,
) -> List[str]:
    """
    Выполняет инференс модели GigaAM v3_e2e_rnnt с помощью ONNX Runtime.

    Параметры
    ----------
    data : Путь к файлу манифеста или итерируемый набор путей к аудио / волновых форм.
    model_cfg : Конфигурация модели.
    sessions : Список инференс-сессий ONNX Runtime (encoder, decoder, joint).
    preprocessor : Optional[FeatureExtractor].
    tokenizer : Optional[Tokenizer].
    batch_size : Размер батча при инференсе.
    num_workers : Количество воркеров для загрузки данных (используйте для больших датасетов).
    progress : Показывать ли прогресс-бар.

    Возвращает
    -------
    List[str]
        Список текстов для каждого сэмпла.
    """
    if preprocessor is None:
        preprocessor = hydra.utils.instantiate(model_cfg.preprocessor)
    if tokenizer is None:
        tokenizer = hydra.utils.instantiate(model_cfg.decoding).tokenizer

    if isinstance(data, str) and Path(data).suffix != ".tsv":
        data = [data]

    dataset = AudioDataset(data)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=num_workers,
    )
    asr_iter = tqdm(loader, desc="ASR inference") if progress else loader

    enc_sess = sessions[0]
    dtype = _session_float_dtype(enc_sess)

    texts = []
    for wavs, wav_lens in asr_iter:
        input_signal, input_lengths = preprocessor(wavs.float(), wav_lens)
        batch_outputs = enc_sess.run(
            [node.name for node in enc_sess.get_outputs()],
            _build_inputs(
                enc_sess,
                [
                    input_signal.contiguous().numpy().astype(dtype),
                    input_lengths.numpy().astype(np.int64),
                ],
            ),
        )

        batch_features = batch_outputs[0]
        assert (
            len(batch_outputs) > 1
        ), "encoder must return enc_lengths for batched decoding"
        batch_lengths = np.asarray(batch_outputs[1], dtype=np.int64).reshape(-1)

        texts.extend(
            _decode_rnnt_batch(
                batch_features, batch_lengths, model_cfg, sessions, tokenizer
            )
        )

    return texts


def _providers_list(provider: Optional[str]) -> List[Union[str, Tuple[str, dict]]]:
    if provider == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    if provider is not None and provider != "CUDAExecutionProvider":
        return [provider]
    cuda_opts = {"cudnn_conv_algo_search": "HEURISTIC"}
    if "CUDAExecutionProvider" in rt.get_available_providers():
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_onnx(
    onnx_dir: str,
    model_version: str,
    provider: Optional[str] = None,
) -> Tuple[List[rt.InferenceSession], omegaconf.DictConfig]:
    """
    Загружает модель GigaAM v3_e2e_rnnt в ONNX Runtime (encoder, decoder, joint).
    """
    providers = _providers_list(provider)

    opts = rt.SessionOptions()
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 8 if providers == ["CPUExecutionProvider"] else 1
    opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3

    model_cfg = omegaconf.OmegaConf.load(f"{onnx_dir}/{model_version}.yaml")
    assert isinstance(model_cfg, omegaconf.DictConfig)

    def _sess(path: str) -> rt.InferenceSession:
        return rt.InferenceSession(path, providers=providers, sess_options=opts)

    pth = f"{onnx_dir}/{model_version}"
    sessions = [
        _sess(f"{pth}_encoder.onnx"),
        _sess(f"{pth}_decoder.onnx"),
        _sess(f"{pth}_joint.onnx"),
    ]

    return sessions, model_cfg
