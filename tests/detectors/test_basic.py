import torch

from streamvigil.detectors import BasicDetector
from tests.mock import MockAutoEncoder

# Mock
auto_encoder = MockAutoEncoder(input_dim=10, hidden_dim=8, latent_dim=5)


def test_predict():
    detector = BasicDetector(auto_encoder)
    x = torch.randn(64, 10)
    scores = detector.predict(x)
    assert ((scores >= 0.0) & (scores <= 1.0)).all().item()
