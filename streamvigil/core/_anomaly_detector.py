from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.optim import Adam, Optimizer


class AnomalyDetector(ABC):
    """
    Anomaly detector base class for model pool based methods.
    """

    def __init__(self, auto_encoder: nn.Module, learning_rate=1e-4) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._learning_rate = learning_rate

        self._auto_encoder = auto_encoder.to(self.device)

    def _load_optimizer(self) -> Optimizer:
        return Adam(self._auto_encoder.parameters(), lr=self._learning_rate)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        _, Z = self._auto_encoder(X)
        return Z

    @abstractmethod
    def stream_train(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def batch_train(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass
