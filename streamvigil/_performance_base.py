from typing import Dict
from uuid import UUID

from torch import Tensor
from torcheval.metrics import BinaryAUPRC

from streamvigil.core import AnomalyDetector, Model, ModelPool


class PerformanceBaseModelPool(ModelPool[Model]):
    def __init__(self, detector: AnomalyDetector, reliability_threshold=0.95, similarity_threshold=0.8) -> None:
        super().__init__(detector, reliability_threshold, similarity_threshold)

    def select_model(self, X: Tensor, y: Tensor) -> UUID:
        """
        Select the model with the highest AUPRC.
        """
        auprc = BinaryAUPRC()

        result: Dict[UUID, Tensor] = {}
        for model in self.get_models():
            scores = model.predict(X)

            auprc.update(scores, y)

            result[model.model_id] = auprc.compute()
            auprc.reset()

        return max(result, key=result.get)
