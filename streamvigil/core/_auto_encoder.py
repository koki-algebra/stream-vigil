from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AutoEncoder(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder: nn.Sequential
        self.decoder: nn.Sequential
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
