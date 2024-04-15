import torch

from streamvigil import ARCUS
from streamvigil.detectors import BasicAutoEncoder, BasicDetector

auto_encoder = BasicAutoEncoder(encoder_dims=[10, 5, 3], decoder_dims=[3, 5, 10])
detector = BasicDetector(auto_encoder)


def test_init():
    x = torch.randn(5, 10)
    arcus = ARCUS(detector)

    # Initialize ARCUS
    arcus.init(x)
    assert len(arcus._pool.get_models()) == 1

    # The model pool does not change even if you reinitialize it.
    arcus.init(x)
    assert len(arcus._pool.get_models()) == 1


def test_run():
    x = torch.randn(5, 10)
    arcus = ARCUS(detector)
    arcus.init(x)

    scores = arcus.run(x)
    assert scores.dim() == 1
    assert scores.numel() == 5
