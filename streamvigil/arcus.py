import uuid

import torch

from streamvigil.core import AnomalyDetector, ModelPool


class ARCUS:
    """
    Adaptive framework foR online deep anomaly deteCtion Under a complex evolving data Stream (ARCUS).
    """

    def __init__(
        self, detector: AnomalyDetector, reliability_threshold=0.95, similarity_threshold=0.8, max_model_num=5
    ) -> None:
        self.pool = ModelPool(detector, reliability_threshold, similarity_threshold, max_model_num)
        self._is_init = False

    def init(self, x: torch.Tensor) -> None:
        if self._is_init:
            return

        # Add initial model
        model_id = self.pool.add_model()
        # Train the new model
        self.pool.train(model_id, x)

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
        # Estimate the anomaly scores
        scores = self.pool.predict(x)

        if self.pool.is_drift():
            # Add new model
            model_id = self.pool.add_model()
            # Train the new model
            self.pool.train(model_id, x)

            # Compress the model pool
            while self.pool.compress(x, model_id):
                continue
        else:
            # Find the most reliable model in the model pool
            model_id = self._find_most_reliable_model()
            # Train the model
            self.pool.train(model_id, x)

        return scores

    def _find_most_reliable_model(self) -> uuid.UUID:
        """
        Find the most reliable model in the model pool.
        """
        max_reliability = -1.0
        models = self.pool.get_models()
        for model in models:
            if max_reliability < model.reliability:
                max_id = model.model_id
                max_reliability = model.reliability

        return max_id
