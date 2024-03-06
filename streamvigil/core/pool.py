import torch


class ModelPool:
    """
    Model Pool.

    Attributes
    ----------
    reliability_threshold : float
        A model pool reliability threshold.
        This threshold should be between 0.0 and 1.0.

    similarity_threshold : float
        A similarity threshold between models.
        This threshold should be between 0.0 and 1.0.
    """

    def __init__(self, reliability_threshold=0.5, similarity_threshold=0.5) -> None:
        if reliability_threshold < 0.0 or reliability_threshold > 1.0:
            raise ValueError("A model pool reliability threshold should be between 0.0 and 1.0")
        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("A similarity threshold should be between 0.0 and 1.0")

        self._reliability_threshold = reliability_threshold
        self._similarity_threshold = similarity_threshold

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run predictions on data matrix x.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.

        Returns
        -------
        y : torch.Tensor
            Anomaly or normal one-hot vector.
        """

        y = torch.randint(0, 2, (x.shape[1], 1), dtype=torch.float64)

        return y

    def add_model(self, x: torch.Tensor) -> None:
        """
        Add a newly initialized model.

        Parameters
        ----------

        Returns
        -------
        """
        pass

    def compress(self) -> None:
        """
        Compress the model pool.

        Parameters
        ----------

        Returns
        -------
        """
        pass
