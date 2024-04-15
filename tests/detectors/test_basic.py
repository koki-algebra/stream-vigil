import uuid

import pytest
import torch
import torch.nn as nn

from streamvigil.core.utils import validate_anomaly_scores
from streamvigil.detectors import BasicAutoEncoder, BasicDetector

auto_encoder = BasicAutoEncoder(encoder_dims=[128, 64, 32, 16], decoder_dims=[16, 32, 64, 128], batch_norm=True)


def test_auto_encoder_forward():
    X = torch.randn(64, 128)
    X_pred = auto_encoder(X)
    criterion = nn.MSELoss()
    criterion(X, X_pred)


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

    valid_scores = torch.rand(128)
    detector.update_reliability(valid_scores)

    invalid_scores = torch.Tensor([-1.0, 0.5, 0.2, 1.5, -0.2])
    with pytest.raises(ValueError):
        detector.update_reliability(invalid_scores)


def test_update_last_batch_scores():
    detector = BasicDetector(auto_encoder)

    valid_scores = torch.rand(128)
    detector.update_last_batch_scores(valid_scores)

    invalid_scores = torch.Tensor([-1.0, 0.5, 0.2, 1.5, -0.2])
    with pytest.raises(ValueError):
        detector.update_last_batch_scores(invalid_scores)


def test_predict():
    detector = BasicDetector(auto_encoder)
    x = torch.randn(64, 128)
    scores = detector.predict(x)
    assert validate_anomaly_scores(scores)
