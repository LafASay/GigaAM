import logging

import pytest
import torch

import gigaam
from gigaam.utils import AudioDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("revision", ["v3_ssl"])
def test_torchaudio_loading(revision, test_audio):
    """Волна, загруженная через torchaudio, должна совпадать с embed_audio(path) (ffmpeg load_audio)."""
    model = gigaam.load_model(revision)
    wav_tns = AudioDataset([test_audio])[0]
    lengths = torch.full([1], wav_tns.shape[-1], device=model._device)
    with torch.no_grad():
        orig_embed = model.embed_audio(test_audio)[0]
        manual_embed, _ = model(
            wav_tns.unsqueeze(0).to(model._device).to(model._dtype), lengths
        )
        diff = (orig_embed - manual_embed).abs().mean().item()
        assert diff < 1e-2, f"Embeddings with torchaudio failed: mean abs diff {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
