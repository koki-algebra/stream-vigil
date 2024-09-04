import copy
from typing import Dict, Generic, List, TypeVar
from uuid import UUID

import torch
from torch import Tensor

from ._anomaly_detector import AnomalyDetector
from ._model import Model
from .similarity import linear_CKA

T = TypeVar("T", bound=Model)


class ARCUSModelPool(Generic[T]):
    def __init__(
        self,
        detector: AnomalyDetector,
        reliability_threshold=0.95,
        similarity_threshold=0.8,
    ) -> None:
        super().__init__()

        self._detector = detector
        self._reliability_threshold = reliability_threshold
        self._similarity_threshold = similarity_threshold

        self._pool: Dict[UUID, T] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_model(self) -> UUID:
        detector = copy.deepcopy(self._detector)
        model = self._create_model(detector)
        self._pool[model.model_id] = model

        return model.model_id

    def _create_model(self, detector: AnomalyDetector) -> T:
        return Model(detector)  # type: ignore

    def get_model(self, model_id: UUID) -> T:
        return self._pool[model_id]

    def get_models(self) -> List[T]:
        return list(self._pool.values())

    def is_drift(self) -> bool:
        """
        Whether concept drift is occurring.
        """
        return self._reliability < self._reliability_threshold

    def similarity(self, X: Tensor, model_id1: UUID, model_id2: UUID) -> float:
        model1 = self.get_model(model_id1)
        model2 = self.get_model(model_id2)
        Z1 = model1.encode(X)
        Z2 = model2.encode(X)

        return linear_CKA(Z1, Z2).item()

    def _merge_models(self, src_id: UUID, dst_id: UUID):
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
        src_params = src_model._detector._auto_encoder.state_dict()
        dst_params = dst_model._detector._auto_encoder.state_dict()

        # Merge parameters
        for key in src_params:
            dst_params[key] = w1 * src_params[key] + w2 * dst_params[key]

        # Load parameters
        self._pool[dst_id]._detector._auto_encoder.load_state_dict(dst_params)

        # Remove source model
        self._pool.pop(src_id)

    def find_most_similar_model(self, X: Tensor, model_id: UUID) -> tuple[UUID, float]:
        """
        Find the model that is most similar to the model with id `model_id`.
        """
        models = self.get_models()
        if len(models) <= 1:
            raise ValueError("No other models exist")

        max_sim = -1.0
        for model in models:
            if model.model_id != model_id:
                sim = self.similarity(X, model_id, model.model_id)
                if max_sim < sim:
                    max_id = model.model_id
                    max_sim = sim

        return max_id, max_sim

    def compress(self, X: Tensor, dst_id: UUID) -> bool:
        """
        Recursively find models similar to the model with id `dst_id` and merge them into destination model.
        """

        if len(self.get_models()) <= 1:
            return False

        # Find the most similar model
        src_id, sim = self.find_most_similar_model(X, dst_id)

        if sim >= self._similarity_threshold:
            self._merge_models(src_id, dst_id)
            return True

        return False

    def stream_train(self, model_id: UUID, X: Tensor) -> Tensor:
        model = self.get_model(model_id)

        # Train the model
        loss = model.stream_train(X)

        # Update the last batch scores
        scores = model.predict(X)
        model.update_last_batch_scores(scores)

        # Increment the number of batches used for training
        model.num_batches += 1

        return loss

    def predict(self, X: Tensor) -> Tensor:
        # Small value to avoid division by zero. Default: 1e-8
        eps = 1e-8

        X = X.to(self.device)

        anomaly_scores = torch.zeros(X.shape[0], device=self.device)
        tmp = 1.0

        for model in self.get_models():
            # Predict the anomaly scores
            scores = model.predict(X)

            # Update the model reliability
            model.update_reliability(scores)

            # Standardized square error
            scores = (scores - scores.mean()) / (scores.std() + eps)

            anomaly_scores += scores * model.reliability

            tmp *= 1 - model.reliability

        # update model pool reliability
        self._reliability = 1 - tmp

        return anomaly_scores
