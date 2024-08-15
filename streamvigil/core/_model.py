import math
import uuid
from abc import ABC

from torch import Tensor

from streamvigil.core import AnomalyDetector


class Model(ABC):
    def __init__(self, detector: AnomalyDetector) -> None:
        self._model_id = uuid.uuid4()
        self._reliability = 0.0

        self._detector = detector

        # The number of batches used to train the model
        self._num_batches = 0

        # Maximum anomaly score on the last batch used to update the model
        self._last_max_score = 0.0
        # Minimum anomaly score on the last batch used to update the model
        self._last_min_score = 0.0
        # Average anomaly score on the last batch used to update the model
        self._last_mean_score = 0.0

    @property
    def model_id(self) -> uuid.UUID:
        return self._model_id

    @property
    def reliability(self) -> float:
        return self._reliability

    def _set_reliability(self, v: float) -> None:
        self._reliability = v

    def update_reliability(self, scores: Tensor):
        if scores.dim() != 1:
            raise ValueError("scores shape must be (1, n).")

        # Small value to avoid division by zero. Default: 1e-8
        eps = 1e-8

        batch_size = scores.numel()
        max_score = max(self._last_max_score, scores.max().item())
        min_score = min(self._last_min_score, scores.min().item())
        gap = abs(self._last_mean_score - scores.mean().item())

        # Model reliability
        reliability = math.exp((-batch_size * gap * gap) / max((max_score - min_score) * (max_score - min_score), eps))

        self._set_reliability(reliability)

    def update_last_batch_scores(self, scores: Tensor) -> None:
        self._last_max_score = scores.max().item()
        self._last_min_score = scores.min().item()
        self._last_mean_score = scores.mean().item()

    @property
    def num_batches(self) -> int:
        """
        The number of batches used for training.
        """
        return self._num_batches

    @num_batches.setter
    def num_batches(self, v: int) -> None:
        if v < 0:
            raise ValueError("The number of batches used for training must be non-negative")
        self._num_batches = v

    def encode(self, X: Tensor) -> Tensor:
        _, Z = self._detector._auto_encoder(X)
        return Z

    def stream_train(self, X: Tensor) -> Tensor:
        return self._detector.stream_train(X)

    def batch_train(self, X: Tensor, y: Tensor) -> Tensor:
        return self._detector.batch_train(X, y)

    def predict(self, X: Tensor) -> Tensor:
        return self._detector.predict(X)
