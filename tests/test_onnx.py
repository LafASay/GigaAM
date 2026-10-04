import logging
import shutil

import numpy as np
import pytest
import torch

import gigaam
from gigaam.onnx_utils import infer_onnx, load_onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "revision, export_dtype",
    [
        ("v3_ssl", torch.float16),
        ("v3_ctc", torch.float16),
        ("v3_e2e_rnnt", torch.float32),
    ],
)
def test_onnx_converting(revision, export_dtype, test_audio):
    """Проверяет конвертацию версии модели в ONNX и корректный батчевый выход."""
    onnx_dir = "test_onnx_tmp"
    model = gigaam.load_model(revision)
    model.to_onnx(dir_path=onnx_dir, dtype=export_dtype)
    sessions, model_cfg = load_onnx(onnx_dir, revision)

    data = [test_audio, test_audio]
    result = infer_onnx(data, model_cfg, sessions, batch_size=2)
    shutil.rmtree(onnx_dir)

    assert isinstance(result, list), f"{revision}: expected list, got {type(result)}"
    assert len(result) == 2, f"{revision}: expected 2 results, got {len(result)}"

    if "ssl" in revision:
        for i, r in enumerate(result):
            assert isinstance(r, np.ndarray), f"{revision}[{i}]: expected ndarray"
            assert r.ndim == 2, f"{revision}[{i}]: expected 2D array, got {r.ndim}D"

        orig_embed = model.embed_audio(test_audio)[0].detach().cpu().numpy()
        tol = 0.01 if export_dtype == torch.float16 else 0.001
        for i in range(2):
            diff = np.abs(orig_embed - result[i]).mean()
            assert diff < tol, f"{revision}[{i}]: ONNX embed diff {diff}"

    else:
        orig_text = model.transcribe(test_audio).text
        for i in range(2):
            assert isinstance(result[i], str), f"{revision}[{i}]: expected str"
            assert (
                orig_text == result[i]
            ), f"{revision}[{i}]: ONNX transcribe failed: {result[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
