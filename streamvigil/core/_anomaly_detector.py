import math
import uuid
from abc import ABC, abstractmethod

import torch
from torch.optim import Adam, Optimizer

from ._auto_encoder import AutoEncoder
from .utils import validate_anomaly_scores


class AnomalyDetector(ABC):
    """
    Anomaly detector base class for model pool based methods.
    """

    def __init__(self, auto_encoder: AutoEncoder, learning_rate=1e-4) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._learning_rate = learning_rate

        self._model_id = uuid.uuid4()
        self._auto_encoder = auto_encoder.to(self.device)
        self._reliability = 0.0

        # Maximum anomaly score on the last batch used to update the model
        self._last_max_score = 0.0
        # Minimum anomaly score on the last batch used to update the model
        self._last_min_score = 0.0
        # Average anomaly score on the last batch used to update the model
        self._last_mean_score = 0.0

    @property
    def model_id(self) -> uuid.UUID:
        """
        model_id : uuid.UUID
            Model ID
        """

        return self._model_id

    @model_id.setter
    def model_id(self, v: uuid.UUID) -> None:
        self._model_id = v

    @property
    def reliability(self) -> float:
        """
        reliability : float
            A model reliability.
            This reliability must be between 0.0 and 1.0.
        """
        return self._reliability

    def _set_reliability(self, v: float) -> None:
        """
        Private setter for reliability property.
        Model reliability must be between 0.0 and 1.0.
        """
        if v < 0.0 or v > 1.0:
            raise ValueError("Model reliability must be between 0.0 and 1.0")
        self._reliability = v

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

        if not validate_anomaly_scores(scores):
            raise ValueError("Anomaly score must be between 0.0 and 1.0")

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
        if not validate_anomaly_scores(scores):
            raise ValueError("Anomaly score must be between 0.0 and 1.0")

        self._last_max_score = scores.max().item()
        self._last_min_score = scores.min().item()
        self._last_mean_score = scores.mean().item()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data matrix `x` into a latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.

        Returns
        -------
        z : torch.Tensor
            Latent representation of `x`.
        """
        x = x.to(self.device)
        return self._auto_encoder.encode(x)

    def _load_optimizer(self) -> Optimizer:
        return Adam(self._auto_encoder.parameters(), lr=self._learning_rate)

    @abstractmethod
    def train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Train the model with unsupervised learning.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.

        Returns
        -------
        loss : torch.Tensor
            Training loss.
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Data matrix.

        Returns
        -------
        scores : torch.Tensor
            Anomaly scores vector. Each score must be between 0.0 and 1.0.
            e.g. Anomaly score for data matrix with 5 samples:
                ```
                scores = torch.Tensor([0.6, 0.2, 0.3, 0.9, 0.1])
                ```
        """
        pass
