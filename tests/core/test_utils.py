import torch

from streamvigil.core.utils import validate_anomaly_scores


def test_validate_anomaly_scores():
    valid_scores = torch.rand(10)
    assert validate_anomaly_scores(valid_scores)

    invalid_scores = torch.Tensor([-1.0, 0.5, 0.2, 1.5, -0.2])
    assert not validate_anomaly_scores(invalid_scores)
