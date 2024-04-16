import uuid
from logging import getLogger

import torch

from streamvigil.core import AnomalyDetector, ModelPool


class ARCUS:
    """
    Adaptive framework foR online deep anomaly deteCtion Under a complex evolving data Stream (ARCUS).
    """

    def __init__(
        self,
        detector: AnomalyDetector,
        reliability_threshold=0.95,
        similarity_threshold=0.8,
        max_epochs=10,
    ) -> None:
        self._pool = ModelPool(
            detector,
            reliability_threshold,
            similarity_threshold,
        )
        self._is_init = False
        self._max_epochs = max_epochs

    def init(self, x: torch.Tensor) -> None:
        if self._is_init:
            return

        logger = getLogger(__name__)

        # Add initial model
        model_id = self._pool.add_model()
        # Train the new model
        for epoch in range(self._max_epochs):
            loss = self._pool.train(model_id, x)
            if epoch % 10 == 0:
                logger.info(f"loss: {loss.item():>7f}")

        self._is_init = True

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the ARCUS algorithm on the data matrix X.

        Parameters
        ----------
        x : torch.Tensor
            Data matrix. The row is the sample sizes and the column is the number of feature dimensions.

        Returns
        -------
        scores : torch.Tensor
            Anomaly scores on the data matrix X.
        """
        logger = getLogger(__name__)

        # Estimate the anomaly scores
        scores = self._pool.predict(x)

        if self._pool.is_drift():
            logger.info("Concept drift detected!")

            # Add new model
            model_id = self._pool.add_model()
            # Train the new model
            logger.info("Start training a new model...")
            for epoch in range(self._max_epochs):
                loss = self._pool.train(model_id, x)
                if epoch % 10 == 0:
                    logger.info(f"loss: {loss.item():>7f}")

            logger.info("Completed training new model!")

            # Compress the model pool
            while self._pool.compress(x, model_id):
                continue
        else:
            # Find the most reliable model in the model pool
            model_id = self._find_most_reliable_model()
            # Train the model
            logger.info(f"Start training model with id {model_id}...")
            for epoch in range(self._max_epochs):
                loss = self._pool.train(model_id, x)
                if epoch % 10 == 0:
                    logger.info(f"loss: {loss.item():>7f}")
            logger.info(f"Completed training model with id {model_id}!")

        return scores

    def _find_most_reliable_model(self) -> uuid.UUID:
        """
        Find the most reliable model in the model pool.
        """
        max_reliability = -1.0
        models = self._pool.get_models()
        for model in models:
            if max_reliability < model.reliability:
                max_id = model.model_id
                max_reliability = model.reliability

        return max_id
