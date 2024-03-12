import math
import uuid
from logging import getLogger

import torch
from torch.nn import MSELoss
from torch.optim import Adam, Optimizer

from streamvigil.core import AutoEncoder


class Model:
    """
    Model.

    Attributes
    ----------
    """

    def __init__(self, auto_encoder: AutoEncoder, max_train_epochs=100) -> None:
        self.__model_id = uuid.uuid4()
        self._auto_encoder = auto_encoder
        self.__reliability = 0.0
        self._max_train_epochs = max_train_epochs

        # Loss function
        self._loss_fn = MSELoss()

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

        return self.__model_id

    @property
    def reliability(self) -> float:
        """
        reliability : float
            A model reliability.
            This reliability must be between 0.0 and 1.0.
        """
        return self.__reliability

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._auto_encoder.encode(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run predictions on data `x`.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.

        Returns
        -------
        scores : torch.Tensor
            Anomaly scores.
        """
        x_pred = self._auto_encoder(x)
        # square error
        scores = (x - x_pred).pow(2).sum(dim=1)

        # estimate the model reliability
        self._update_reliability(scores)

        return scores

    def _update_reliability(self, scores: torch.Tensor) -> None:
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

        batch_size = scores.numel()
        max_score = max(self._last_max_score, scores.max().item())
        min_score = min(self._last_min_score, scores.min().item())
        gap = abs(self._last_mean_score - scores.mean().item())

        self.__reliability = math.exp((-batch_size * gap * gap) / ((max_score - min_score) * (max_score - min_score)))

    def _load_optimizer(self) -> Optimizer:
        optimizer = Adam(self._auto_encoder.parameters())
        return optimizer

    def train(self, x: torch.Tensor):
        """
        Train on data matrix x.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.
        """
        logger = getLogger(__name__)

        # Optimizer
        optimizer = self._load_optimizer()

        # Training the model
        self._auto_encoder.train()
        for epoch in range(self._max_train_epochs):
            logger.info("Epoch {}".format(epoch))

            # Compute prediction and loss
            x_pred = self._auto_encoder(x)
            loss = self._loss_fn(x_pred, x)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Update last batch scores
        scores = self.predict(x)
        self._last_max_score = scores.max().item()
        self._last_min_score = scores.min().item()
        self._last_mean_score = scores.mean().item()
