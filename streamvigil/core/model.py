import torch


class Model:
    """
    Model Pool.

    Attributes
    ----------
    """

    def __init__(self) -> None:
        self.__reliability = 0.0

    @property
    def reliability(self):
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
        return x

    def train(self, x: torch.Tensor):
        """
        Train on data matrix x.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.
        """
        pass
