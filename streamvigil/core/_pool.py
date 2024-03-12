import uuid
from typing import Dict, List

import torch

from streamvigil.core import AutoEncoder, Model
from streamvigil.core._similarity import linear_CKA


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

    def get_model(self, model_id: uuid.UUID) -> Model:
        """
        Get the model with `model_id`.
        """
        return self._pool[model_id]

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

        for model in self.get_models():
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

        if len(self.get_models()) >= self._max_model_num:
            raise ValueError("The maximum number of models in the model pool is {}".format(self._max_model_num))

        # initialize new model
        model = Model(self._auto_encoder)

        # add new model to model pool
        self._pool[model.model_id] = model

        return model.model_id

    def similarity(self, x: torch.Tensor, model_id1: uuid.UUID, model_id2: uuid.UUID) -> float:
        """
        Calculate the similarity between model ID1 and ID2.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix

        model_id1, model_id2 : uuid.UUID
            IDs for which you want to calculate similarity.

        Returns
        -------
        similarity : float
            Model similarity.
        """
        z1 = self.get_model(model_id1).encode(x)
        z2 = self.get_model(model_id2).encode(x)

        return linear_CKA(z1, z2).item()

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

    def _merge_models(self, src_id: uuid.UUID, dst_id: uuid.UUID) -> None:
        """
        Merge source model into destination model.
        """
        # Weghts
        w1 = 0.5
        w2 = 0.5

        # Model parameters
        src_params = self.get_model(src_id)._auto_encoder.state_dict()
        dst_params = self.get_model(dst_id)._auto_encoder.state_dict()

        # Merge parameters
        for key in src_params:
            dst_params[key] = w1 * src_params[key] + w2 * dst_params[key]

        # Load parameters
        self._pool[dst_id]._auto_encoder.load_state_dict(dst_params)

        # Remove source model
        self._pool.pop(src_id)

    def compress(self, x: torch.Tensor, target_model_id: uuid.UUID) -> None:
        """
        Compress the model pool.

        Parameters
        ----------

        Returns
        -------
        """
        pass
