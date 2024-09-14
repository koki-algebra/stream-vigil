import copy
from typing import Dict, Generic, List, TypeVar
from uuid import UUID

import torch
from torch import Tensor

from ._anomaly_detector import AnomalyDetector
from ._model import Model
from .similarity import linear_CKA

T = TypeVar("T", bound=Model)


class ModelPool(Generic[T]):
    def __init__(
        self,
        detector: AnomalyDetector,
        similarity_threshold=0.8,
        historical_window_size=10000,
        latest_window_size=10000,
        last_trained_size=10000,
        drift_alpha=0.05,
        adapted_alpha=0.05,
    ) -> None:
        self._detector = detector
        self._reliability = 1.0
        self._similarity_threshold = similarity_threshold

        self._historical_window_size = historical_window_size
        self._latest_window_size = latest_window_size
        self._last_trained_size = last_trained_size
        self._drift_alpha = drift_alpha
        self._adapted_alpha = adapted_alpha

        self._pool: Dict[UUID, T] = {}

        self._current_model_id = self.add_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_model(self) -> UUID:
        detector = copy.deepcopy(self._detector)
        model = self._create_model(detector)
        self._pool[model.model_id] = model

        return model.model_id

    def _create_model(self, detector: AnomalyDetector) -> T:
        return Model(
            detector,
            historical_window_size=self._historical_window_size,
            latest_window_size=self._latest_window_size,
            last_trained_size=self._last_trained_size,
            drift_alpha=self._drift_alpha,
            adapted_alpha=self._adapted_alpha,
        )  # type: ignore

    def get_model(self, model_id: UUID) -> T:
        return self._pool[model_id]

    def get_models(self) -> List[T]:
        return list(self._pool.values())

    @property
    def current_model_id(self):
        return self._current_model_id

    @current_model_id.setter
    def current_model_id(self, model_id: UUID):
        if model_id not in self._pool:
            raise ValueError(f"model_id {model_id} does not exist in the model pool")

        self._current_model_id = model_id

    def find_adapted_model(self) -> UUID | None:
        for model in self.get_models():
            if model.model_id == self.current_model_id:
                continue

            if model.is_adapted():
                return model.model_id

        return None

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

    def stream_train(self, X: Tensor) -> Tensor:
        current_model = self.get_model(self.current_model_id)

        # Train the model
        loss = current_model.stream_train(X)

        # Update last trained window
        scores = current_model.predict(X)
        current_model.last_trained_window.push(scores)

        # Increment the number of batches used for training
        current_model.num_batches += 1

        return loss

    def batch_train(self, model_id: UUID, X: Tensor, y: Tensor) -> Tensor:
        model = self.get_model(model_id)
        loss = model.batch_train(X, y)

        return loss

    def predict(self, X: Tensor) -> Tensor:
        # Update latest window
        for model in self.get_models():
            scores = model.predict(X)
            model.latest_window.push(scores)

        current_model = self.get_model(self.current_model_id)

        return current_model.predict(X)
