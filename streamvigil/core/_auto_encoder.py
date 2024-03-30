from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AutoEncoder(ABC, nn.Module):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
