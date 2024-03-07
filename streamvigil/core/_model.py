import math
import uuid

import torch

from streamvigil.core import AutoEncoder


class Model:
    """
    Model.

    Attributes
    ----------
    """

    def __init__(self, auto_encoder: AutoEncoder) -> None:
        self.__model_id = uuid.uuid4()
        self._auto_encoder = auto_encoder
        self.__reliability = 0.0
        self.__last_max_score = 0.0
        self.__last_min_score = 0.0
        self.__last_mean_score = 0.0

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

    @reliability.setter
    def reliability(self, v: float):
        if v < 0.0 or v > 1.0:
            raise ValueError("A model reliability must be between 0.0 and 1.0")
        self.__reliability = v

    @property
    def last_max_score(self) -> float:
        """
        last_max_score : float
            Maximum anomaly score on the last batch used to update the model.
        """
        return self.__last_max_score

    @property
    def last_min_score(self) -> float:
        """
        last_min_score : float
            Minimum anomaly score on the last batch used to update the model.
        """
        return self.__last_min_score

    @property
    def last_mean_score(self) -> float:
        """
        last_mean_score : float
            Average anomaly score on the last batch used to update the model.
        """
        return self.__last_mean_score

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
        x_pred = self._auto_encoder.forward(x)
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
        if len(scores) != 2 or scores.shape[0] != 1:
            raise ValueError("scores shape must be (1, n).")

        batch_size = scores.shape[1]
        max_score = max(self.last_max_score, scores.max().item())
        min_score = min(self.last_min_score, scores.min().item())
        gap = abs(self.last_mean_score - scores.mean().item())

        self.reliability = math.exp((-batch_size * gap * gap) / ((max_score - min_score) * (max_score - min_score)))

    def train(self, x: torch.Tensor):
        """
        Train on data matrix x.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.
        """
        # Todo: training the model

        # Update last batch scores
        scores = self.predict(x)
        self.__last_max_score = scores.max().item()
        self.__last_min_score = scores.min().item()
        self.__last_mean_score = scores.mean().item()
