import uuid

from scipy import stats
from torch import Tensor

from ._anomaly_detector import AnomalyDetector
from ._window import Window


class Model:
    def __init__(
        self,
        detector: AnomalyDetector,
        historical_window_size=1000,
        latest_window_size=1000,
        last_trained_size=1000,
        alpha=0.05,
    ) -> None:
        self._model_id = uuid.uuid4()
        self._detector = detector

        # The number of batches used to train the model
        self._num_batches = 0

        # Historical window
        self.historical_window = Window(historical_window_size)

        # Latest window
        self.latest_window = Window(latest_window_size)

        # Last trained window
        self.last_trained_window = Window(last_trained_size)

        self._alpha = alpha

    @property
    def model_id(self) -> uuid.UUID:
        return self._model_id

    @property
    def num_batches(self) -> int:
        """
        The number of batches used for training.
        """
        return self._num_batches

    @num_batches.setter
    def num_batches(self, v: int) -> None:
        if v < 0:
            raise ValueError("The number of batches used for training must be non-negative")
        self._num_batches = v

    def is_drift(self) -> bool:
        """
        Test whether the historical window and the latest window are significantly different.
        If there is a significant difference between them, it is considered that concept drift is occurring.
        """
        if (
            len(self.historical_window) < self.historical_window.max_size
            or len(self.latest_window) < self.latest_window.max_size
        ):
            return False

        _, p_value = stats.mannwhitneyu(
            self.latest_window.get_items(),
            self.historical_window.get_items(),
            alternative="greater",
        )

        return p_value < self._alpha

    def is_adapted(self) -> bool:
        """
        Test whether the last trained window and the latest window are significantly different.
        If there is no significant difference, it is considered to have been adapted.
        """
        if (
            len(self.last_trained_window) < self.last_trained_window.max_size
            or len(self.latest_window) < self.latest_window.max_size
        ):
            return False

        _, p_value = stats.mannwhitneyu(
            self.latest_window.get_items(),
            self.last_trained_window.get_items(),
            alternative="greater",
        )

        return p_value >= self._alpha

    def encode(self, X: Tensor) -> Tensor:
        return self._detector.encode(X)

    def stream_train(self, X: Tensor) -> Tensor:
        return self._detector.stream_train(X)

    def batch_train(self, X: Tensor, y: Tensor) -> Tensor:
        return self._detector.batch_train(X, y)

    def predict(self, X: Tensor) -> Tensor:
        return self._detector.predict(X)
