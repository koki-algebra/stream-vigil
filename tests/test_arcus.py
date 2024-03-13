import torch

from streamvigil import ARCUS
from tests.mock import MockAutoEncoder

# Mock
auto_encoder = MockAutoEncoder(input_dim=10, hidden_dim=8, latent_dim=5)


def test_init():
    x = torch.randn(5, 10)
    arcus = ARCUS(auto_encoder)

    # Initialize ARCUS
    arcus.init(x)
    assert len(arcus.pool.get_models()) == 1

    # The model pool does not change even if you reinitialize it.
    arcus.init(x)
    assert len(arcus.pool.get_models()) == 1


def test_run():
    x = torch.randn(5, 10)
    arcus = ARCUS(auto_encoder)
    arcus.init(x)

    arcus.run(x)
