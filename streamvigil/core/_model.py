import uuid

from torch import Tensor

from streamvigil.core import AnomalyDetector


class Model:
    def __init__(self, detector: AnomalyDetector) -> None:
        self._model_id = uuid.uuid4()
        self._detector = detector

        # The number of batches used to train the model
        self._num_batches = 0

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
        pass

    def encode(self, X: Tensor) -> Tensor:
        return self._detector.encode(X)

    def stream_train(self, X: Tensor) -> Tensor:
        return self._detector.stream_train(X)

    def batch_train(self, X: Tensor, y: Tensor) -> Tensor:
        return self._detector.batch_train(X, y)

    def predict(self, X: Tensor) -> Tensor:
        return self._detector.predict(X)
