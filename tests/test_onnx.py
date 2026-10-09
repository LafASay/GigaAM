import logging
import shutil

import pytest
import torch

import gigaam
from gigaam.onnx_utils import infer_onnx, load_onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "revision, export_dtype",
    [
        ("v3_e2e_rnnt", torch.float32),
    ],
)
def test_onnx_converting(revision, export_dtype, test_audio):
    """Проверяет конвертацию модели в ONNX и корректный батчевый выход."""
    onnx_dir = "test_onnx_tmp"
    model = gigaam.load_model(revision)
    model.to_onnx(dir_path=onnx_dir, dtype=export_dtype)
    sessions, model_cfg = load_onnx(onnx_dir, revision)

    data = [test_audio, test_audio]
    result = infer_onnx(data, model_cfg, sessions, batch_size=2)
    shutil.rmtree(onnx_dir)

    assert isinstance(result, list), f"{revision}: expected list, got {type(result)}"
    assert len(result) == 2, f"{revision}: expected 2 results, got {len(result)}"

    orig_text = model.transcribe(test_audio).text
    for i in range(2):
        assert isinstance(result[i], str), f"{revision}[{i}]: expected str"
        assert (
            orig_text == result[i]
        ), f"{revision}[{i}]: ONNX transcribe failed: {result[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
