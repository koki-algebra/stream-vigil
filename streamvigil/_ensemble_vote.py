from typing import Dict
from uuid import UUID

import torch
from torch import Tensor

from streamvigil.core import AnomalyDetector, Model, ModelPool


class EnsembleVoteModelPool(ModelPool[Model]):
    def __init__(self, detector: AnomalyDetector, reliability_threshold=0.95, similarity_threshold=0.8) -> None:
        super().__init__(detector, reliability_threshold, similarity_threshold)

    def select_model(self, X: Tensor) -> UUID:
        """
        Select the model that is closest to the prediction result of the model pool.
        """
        pool_scores = self.predict(X)

        result: Dict[UUID, Tensor] = {}
        for model in self.get_models():
            model_scores = model.predict(X)

            # Mean Squared Error (MSE)
            result[model.model_id] = torch.mean((model_scores - pool_scores) ** 2)

        return min(result, key=lambda k: result[k].item())
