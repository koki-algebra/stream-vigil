import torch

from streamvigil import ARCUS
from streamvigil.detectors import BasicDetector
from tests.mock import MockAutoEncoder

# Mock
auto_encoder = MockAutoEncoder(input_dim=10, hidden_dim=8, latent_dim=5)
detector = BasicDetector(auto_encoder)


def test_init():
    x = torch.randn(5, 10)
    arcus = ARCUS(detector)

    # Initialize ARCUS
    arcus.init(x)
    assert len(arcus.pool.get_models()) == 1

    # The model pool does not change even if you reinitialize it.
    arcus.init(x)
    assert len(arcus.pool.get_models()) == 1


def test_run():
    x = torch.randn(5, 10)
    arcus = ARCUS(detector)
    arcus.init(x)

    scores = arcus.run(x)
    assert scores.dim() == 1
    assert scores.numel() == 5
