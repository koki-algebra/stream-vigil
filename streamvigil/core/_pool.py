import uuid
from typing import Dict, List

import torch

from streamvigil.core import AutoEncoder, Model


class ModelPool:
    """
    Model Pool.

    Attributes
    ----------
    reliability_threshold : float
        A model pool reliability threshold.
        This threshold must be between 0.0 and 1.0.

    similarity_threshold : float
        A similarity threshold between models.
        This threshold must be between 0.0 and 1.0.
    """

    def __init__(
        self, auto_encoder: AutoEncoder, reliability_threshold=0.5, similarity_threshold=0.5, max_model_num=5
    ) -> None:
        if reliability_threshold < 0.0 or reliability_threshold > 1.0:
            raise ValueError("A model pool reliability threshold must be between 0.0 and 1.0")
        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("A similarity threshold must be between 0.0 and 1.0")

        self._reliability_threshold = reliability_threshold
        self._similarity_threshold = similarity_threshold
        self._max_model_num = max_model_num
        self._auto_encoder = auto_encoder

        # model pool reliability
        self._reliability = 0.0
        # model pool
        self._pool: Dict[uuid.UUID, Model] = {}

    def is_drift(self) -> bool:
        """
        Whether concept drift is occurring.
        """
        return self._reliability < self._reliability_threshold

    def get_models(self) -> List[Model]:
        """
        Get a list of models included in the model pool.

        Returns
        -------
        models : List[Model]
        """

        return list(self._pool.values())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run predictions on data matrix `x`.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix.

        Returns
        -------
        anomaly_scores : torch.Tensor
            Anomaly scores.
        """
        anomaly_scores = torch.zeros(1, x.shape[0])
        tmp = 1.0

        for model in self._pool.values():
            scores = model.predict(x)

            # standardized square error
            scores = (scores - scores.mean()) / scores.std()

            anomaly_scores += scores * model.reliability

            tmp *= 1 - model.reliability

        # update model pool reliability
        self._reliability = 1 - tmp

        return anomaly_scores

    def add_model(self) -> uuid.UUID:
        """
        Add a newly initialized model.
        You cannot add more models than the maximum number of models.

        Parameters
        ----------

        Returns
        -------
        model_id : uuid.UUID
            ID of newly added model.
        """

        if len(self._pool) >= self._max_model_num:
            raise ValueError("The maximum number of models in the model pool is {}".format(self._max_model_num))

        # initialize new model
        model = Model(self._auto_encoder)

        # add new model to model pool
        self._pool[model.model_id] = model

        return model.model_id

    def train(self, model_id: uuid.UUID, x: torch.Tensor) -> None:
        """
        Train the model with `model_id` with data matrix `x`

        Parameters
        ----------
        model_id : uuid.UUID
            ID of the model to be trained.

        x : torch.Tensor
            Data matrix

        Returns
        -------
        """
        self._pool[model_id].train(x)

    def compress(self) -> None:
        """
        Compress the model pool.

        Parameters
        ----------

        Returns
        -------
        """
        pass
