import copy
import uuid
from typing import Dict, List

import torch

from ._anomaly_detector import AnomalyDetector
from .similarity import linear_CKA


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

    def __init__(self, detector: AnomalyDetector, reliability_threshold=0.95, similarity_threshold=0.8) -> None:
        if reliability_threshold < 0.0 or reliability_threshold > 1.0:
            raise ValueError("A model pool reliability threshold must be between 0.0 and 1.0")
        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("A similarity threshold must be between 0.0 and 1.0")

        self._reliability_threshold = reliability_threshold
        self._similarity_threshold = similarity_threshold
        self._detector = detector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model pool reliability
        self._reliability = 0.0
        # model pool
        self._pool: Dict[uuid.UUID, AnomalyDetector] = {}

    def is_drift(self) -> bool:
        """
        Whether concept drift is occurring.
        """
        return self._reliability < self._reliability_threshold

    def get_models(self) -> List[AnomalyDetector]:
        """
        Get a list of models included in the model pool.

        Returns
        -------
        models : List[Model]
        """

        return list(self._pool.values())

    def get_model(self, model_id: uuid.UUID) -> AnomalyDetector:
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
        # Small value to avoid division by zero. Default: 1e-8
        eps = 1e-8

        x = x.to(self.device)

        anomaly_scores = torch.zeros(x.shape[0]).to(self.device)
        tmp = 1.0

        for model in self.get_models():
            # Predict the anomaly scores
            scores = model.predict(x)

            # Update the model reliability
            model.update_reliability(scores)

            # Standardized square error
            scores = (scores - scores.mean()) / (scores.std() + eps)

            anomaly_scores += scores * model.reliability

            tmp *= 1 - model.reliability

        # update model pool reliability
        self._reliability = 1 - tmp

        return anomaly_scores.sigmoid()

    def add_model(self) -> uuid.UUID:
        """
        Add a newly initialized model.

        Parameters
        ----------

        Returns
        -------
        model_id : uuid.UUID
            ID of newly added model.
        """
        # initialize new model
        detector = copy.deepcopy(self._detector)
        detector.model_id = uuid.uuid4()

        # add new model to model pool
        self._pool[detector.model_id] = detector

        return detector.model_id

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

    def train(self, model_id: uuid.UUID, x: torch.Tensor) -> torch.Tensor:
        """
        Train the model with `model_id` with data matrix `x`

        Parameters
        ----------
        model_id : uuid.UUID
            ID of the model to be trained.

        x : torch.Tensor
            Data matrix.

        Returns
        -------
        loss : torch.Tensor
            Training loss.
        """
        model = self.get_model(model_id)
        # Train the model
        loss = model.train(x)

        # Update the last batch scores
        scores = model.predict(x)
        model.update_last_batch_scores(scores)

        # Increment the number of batches used for training
        model.num_batches += 1

        return loss

    def _merge_models(self, src_id: uuid.UUID, dst_id: uuid.UUID) -> None:
        """
        Merge source model into destination model.
        """
        src_model = self.get_model(src_id)
        dst_model = self.get_model(dst_id)

        # Weghts
        w1 = 0.5
        w2 = 0.5
        num_batches = src_model.num_batches + dst_model.num_batches
        if num_batches != 0:
            w1 = src_model.num_batches / num_batches
            w2 = dst_model.num_batches / num_batches

        # Model parameters
        src_params = src_model._auto_encoder.state_dict()
        dst_params = dst_model._auto_encoder.state_dict()

        # Merge parameters
        for key in src_params:
            dst_params[key] = w1 * src_params[key] + w2 * dst_params[key]

        # Load parameters
        self._pool[dst_id]._auto_encoder.load_state_dict(dst_params)

        # Remove source model
        self._pool.pop(src_id)

    def find_most_similar_model(self, x: torch.Tensor, model_id: uuid.UUID) -> tuple[uuid.UUID, float]:
        models = self.get_models()
        if len(models) <= 1:
            raise ValueError("No other models exist")

        max_sim = -1.0
        for model in models:
            if model.model_id != model_id:
                sim = self.similarity(x, model_id, model.model_id)
                if max_sim < sim:
                    max_id = model.model_id
                    max_sim = sim

        return max_id, max_sim

    def compress(self, x: torch.Tensor, dst_id: uuid.UUID) -> bool:
        """
        Compress the model pool.

        Parameters
        ----------
        dst_id : uuid.UUID
            Destination model ID.
            Other models will be merged into this model.

        Returns
        -------
        is_compressed : bool
            Whether model pool compression was performed.
        """
        if len(self.get_models()) <= 1:
            return False

        # Find the most similar model
        src_id, sim = self.find_most_similar_model(x, dst_id)

        if sim >= self._similarity_threshold:
            self._merge_models(src_id, dst_id)
            return True

        return False
