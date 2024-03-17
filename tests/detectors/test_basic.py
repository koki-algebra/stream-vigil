import uuid

import pytest
import torch

from streamvigil.core.utils import validate_anomaly_scores
from streamvigil.detectors import BasicDetector
from tests.mock import MockAutoEncoder

# Mock
auto_encoder = MockAutoEncoder(input_dim=10, hidden_dim=8, latent_dim=5)


def test_model_id():
    detector = BasicDetector(auto_encoder)

    assert detector.model_id is not None
    assert isinstance(detector.model_id, uuid.UUID)

    # set new id
    detector.model_id = uuid.uuid4()


def test_reliability():
    detector = BasicDetector(auto_encoder)
    with pytest.raises(ValueError):
        detector._set_reliability(-1.5)
    with pytest.raises(ValueError):
        detector._set_reliability(1.5)


def test_update_reliability():
    detector = BasicDetector(auto_encoder)

    valid_scores = torch.rand(10)
    detector.update_reliability(valid_scores)

    invalid_scores = torch.Tensor([-1.0, 0.5, 0.2, 1.5, -0.2])
    with pytest.raises(ValueError):
        detector.update_reliability(invalid_scores)


def test_update_last_batch_scores():
    detector = BasicDetector(auto_encoder)

    valid_scores = torch.rand(10)
    detector.update_last_batch_scores(valid_scores)

    invalid_scores = torch.Tensor([-1.0, 0.5, 0.2, 1.5, -0.2])
    with pytest.raises(ValueError):
        detector.update_last_batch_scores(invalid_scores)


def test_predict():
    detector = BasicDetector(auto_encoder)
    x = torch.randn(64, 10)
    scores = detector.predict(x)
    assert validate_anomaly_scores(scores)
