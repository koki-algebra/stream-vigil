from typing import override
from uuid import UUID, uuid4

from scipy.stats import gaussian_kde
from torch import Tensor

from streamvigil.core import AnomalyDetector, Model, ModelPool


class ModelWithKDE(Model):
    def __init__(self, detector: AnomalyDetector) -> None:
        super().__init__(detector)

        self.kde: gaussian_kde | None = None

    @override
    def stream_train(self, X: Tensor) -> Tensor:
        self.kde = gaussian_kde(X.T.cpu().numpy())
        return super().stream_train(X)


class ModelPoolWithKDE(ModelPool[ModelWithKDE]):
    def __init__(
        self,
        detector: AnomalyDetector,
        reliability_threshold=0.95,
        similarity_threshold=0.8,
    ) -> None:
        super().__init__(detector, reliability_threshold, similarity_threshold)

    @override
    def _create_model(self, detector: AnomalyDetector) -> ModelWithKDE:
        return ModelWithKDE(detector)

    def select_model(self, X_feedback: Tensor) -> UUID:
        # TODO: implement
        return uuid4()
