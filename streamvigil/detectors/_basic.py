from typing import List

import torch
import torch.nn as nn
from torch.nn import MSELoss

from streamvigil.core import AnomalyDetector, AutoEncoder


class BasicAutoEncoder(AutoEncoder):
    def __init__(self, encoder_dims: List[int], decoder_dims: List[int], batch_norm=False) -> None:
        super().__init__()

        # Build network
        self.encoder = self._build_network(encoder_dims, batch_norm)
        self.decoder = self._build_network(decoder_dims, batch_norm)

    def _build_network(self, dims: List[int], batch_norm=False):
        network = nn.Sequential()
        for i, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            network.append(nn.Linear(input_dim, output_dim))
            if i != len(dims) - 2:
                if batch_norm:
                    network.append(nn.BatchNorm1d(output_dim))
                network.append(nn.ReLU())

        return network

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class BasicDetector(AnomalyDetector):
    def __init__(self, auto_encoder: AutoEncoder, learning_rate=0.0001) -> None:
        super().__init__(auto_encoder, learning_rate)

        # Loss function
        self._criterion = MSELoss()

    def train(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        # Optimizer
        optimizer = self._load_optimizer()

        # Training the model
        self._auto_encoder.train()
        # Compute prediction and loss
        x_pred: torch.Tensor = self._auto_encoder(x)
        loss: torch.Tensor = self._criterion(x_pred, x)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self._auto_encoder.eval()

        x = x.to(self.device)
        x_pred: torch.Tensor = self._auto_encoder(x)

        # Square error
        errs = (x - x_pred).pow(2).sum(dim=1)

        # Anomaly scores
        scores = errs.sigmoid()

        return scores
