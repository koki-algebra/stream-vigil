from typing import Dict
from uuid import UUID

import torch
from torch import Tensor

from streamvigil.core import AnomalyDetector, Model, ModelPool


class PerformanceBaseModelPool(ModelPool[Model]):
    def __init__(self, detector: AnomalyDetector, reliability_threshold=0.95, similarity_threshold=0.8) -> None:
        super().__init__(detector, reliability_threshold, similarity_threshold)

    def select_model(self, X: Tensor, y: Tensor) -> UUID:
        """
        Select the model with the smallest reconstruction error.
        """
        normal_indices = torch.where(y == 0)[0]
        X_normal = X[normal_indices]

        result: Dict[UUID, Tensor] = {}
        for model in self.get_models():
            scores = model.predict(X_normal)
            result[model.model_id] = scores.sum()

        return min(result, key=lambda k: result[k].item())
