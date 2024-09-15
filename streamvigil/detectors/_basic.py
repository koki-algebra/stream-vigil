from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import MSELoss

from streamvigil.core import AnomalyDetector


class BasicAutoEncoder(nn.Module):
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

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Z = self.encoder(X)
        X_pred = self.decoder(Z)
        return X_pred, Z


class BasicDetector(AnomalyDetector):
    def __init__(
        self,
        auto_encoder: nn.Module,
        learning_rate=0.0001,
    ) -> None:
        super().__init__(auto_encoder, learning_rate)

        # Loss function
        self._criterion = MSELoss()

    def stream_train(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)

        # Optimizer
        optimizer = self._load_optimizer()

        # Training the model
        self._auto_encoder.train()
        # Compute prediction and loss
        X_pred, _ = self._auto_encoder(X)
        loss: torch.Tensor = self._criterion(X_pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def batch_train(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        y = y.to(self.device)

        optimizer = self._load_optimizer()
        criterion = nn.MSELoss(reduction="none")

        X_pred, _ = self._auto_encoder(X)
        losses: torch.Tensor = criterion(X_pred, X).mean(dim=1)

        normal_losses = losses[torch.logical_or(y == 0, y.isnan())]
        anomaly_losses = losses[y == 1]

        if len(anomaly_losses) == 0:
            anomaly_loss = torch.tensor(0.0, requires_grad=True)
        else:
            anomaly_loss = anomaly_losses.mean()

        if len(normal_losses) == 0:
            normal_loss = torch.tensor(0.0, requires_grad=True)
        else:
            normal_loss = normal_losses.mean()

        loss = normal_loss - anomaly_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self._auto_encoder.eval()

        X = X.to(self.device)
        X_pred, _ = self._auto_encoder(X)

        return (X - X_pred).pow(2).mean(dim=1)
