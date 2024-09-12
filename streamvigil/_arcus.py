from logging import getLogger
from uuid import UUID
from torch import Tensor

from ._arcus_model_pool import ARCUSModelPool
from .core._anomaly_detector import AnomalyDetector
from .core._model import Model


class ARCUS:
    def __init__(
        self,
        detector: AnomalyDetector,
        logger=getLogger(__name__),
        reliability_threshold=0.95,
        similarity_threshold=0.8,
    ) -> None:
        self.logger = logger
        self.model_pool = ARCUSModelPool[Model](
            detector,
            reliability_threshold,
            similarity_threshold,
        )

    def init(self) -> UUID:
        # Add initial model
        return self.model_pool.add_model()

    def stream_train(self, X: Tensor, is_logging=False):
        if self.model_pool.is_drift():
            self.logger.info("concept drift detected!")

            # Add new model and train it
            model_id = self.model_pool.add_model()
            self.logger.info(f"new model {model_id} is added")

            self.model_pool.stream_train(model_id, X)

            # Compress model pool
            if self.model_pool.compress(X, model_id):
                self.logger.info("model pool compressed!")
        else:
            # Update the most reliable model
            model_id = self.model_pool.find_most_reliable_model()
            loss = self.model_pool.stream_train(model_id, X)

            if is_logging:
                self.logger.info(f"model {model_id} is selected: loss = {loss.item():0.5f}")

    def update_reliability(self, X: Tensor, is_logging=False):
        if is_logging:
            self.logger.info(f"[before] model pool reliability = {self.model_pool._reliability:0.5f}")

        self.model_pool.update_reliability(X)

        if is_logging:
            self.logger.info(f"[after] model pool reliability = {self.model_pool._reliability:0.5f}")

    def predict(self, X: Tensor) -> Tensor:
        return self.model_pool.predict(X)
