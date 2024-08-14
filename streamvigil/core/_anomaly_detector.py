import math
import uuid
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.optim import Adam, Optimizer


class AnomalyDetector(ABC):
    """
    Anomaly detector base class for model pool based methods.
    """

    def __init__(self, auto_encoder: nn.Module, learning_rate=1e-4) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._learning_rate = learning_rate

        self._model_id = uuid.uuid4()
        self._auto_encoder = auto_encoder.to(self.device)
        self._reliability = 0.0

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

    @model_id.setter
    def model_id(self, v: uuid.UUID) -> None:
        self._model_id = v

    @property
    def reliability(self) -> float:
        return self._reliability

    def _set_reliability(self, v: float) -> None:
        self._reliability = v

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

    def update_reliability(self, scores: torch.Tensor) -> None:
        """
        Update model reliability based on Hoeffding's bound.

        Parameters
        ----------
        scores : torch.Tensor
            Anomaly scores.

        Returns
        -------
        """
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

    def update_last_batch_scores(self, scores: torch.Tensor) -> None:
        self._last_max_score = scores.max().item()
        self._last_min_score = scores.min().item()
        self._last_mean_score = scores.mean().item()

    def _load_optimizer(self) -> Optimizer:
        return Adam(self._auto_encoder.parameters(), lr=self._learning_rate)

    @abstractmethod
    def stream_train(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def batch_train(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass
