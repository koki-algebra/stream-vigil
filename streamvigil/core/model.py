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
        self.__reliability = 0.0
        self._auto_encoder = auto_encoder

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
            This reliability should be between 0.0 and 1.0.
        """

        return self.__reliability

    @reliability.setter
    def reliability(self, v: float):
        if v < 0.0 or v > 1.0:
            raise ValueError("A model reliability should be between 0.0 and 1.0")
        self.__reliability = v

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run predictions on data x.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.

        Returns
        -------
        x_pred : torch.Tensor
            Data matrix reconstructed by autoencoder.
        """

        return self._auto_encoder.forward(x)

    def train(self, x: torch.Tensor):
        """
        Train on data matrix x.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.
        """
        pass
