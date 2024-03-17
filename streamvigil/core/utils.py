import torch


def validate_anomaly_scores(scores: torch.Tensor) -> bool:
    return ((scores >= 0.0) & (scores <= 1.0)).all().item() is True
