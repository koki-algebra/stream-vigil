from typing import List, override

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

from streamvigil.core import AnomalyDetector, AutoEncoder


class RSR(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.A)


class _RSRAE(AutoEncoder):
    """
    Robust Subspace Recovery Auto Encoder (RSRAE)
    """

    def __init__(self, encoder_dims: List[int], decoder_dims: List[int], rsr_dim: int) -> None:
        super().__init__()

        input_dim = encoder_dims[0]

        # Encoder
        self.encoder = nn.Sequential()
        for output_dim in encoder_dims:
            self.encoder.append(nn.Linear(input_dim, output_dim))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        # Robust Subspace Recovery layer (RSR layer)
        self.rsr = RSR(encoder_dims[-1], rsr_dim)

        # Decoder
        input_dim = rsr_dim
        self.decoder = nn.Sequential()
        for i, output_dim in enumerate(decoder_dims):
            self.decoder.append(nn.Linear(input_dim, output_dim))
            if i != len(decoder_dims) - 1:
                self.decoder.append(nn.ReLU())
                self.decoder.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z_tilde = self.rsr(z)
        return self.decode(z_tilde)


class RSRAE(AnomalyDetector):
    def __init__(
        self,
        encoder_dims: List[int],
        decoder_dims: List[int],
        rsr_dim: int,
        lambda1=1.0,
        lambda2=1.0,
        learning_rate=0.0001,
    ) -> None:
        auto_encoder = _RSRAE(encoder_dims, decoder_dims, rsr_dim)
        super().__init__(auto_encoder, learning_rate)

        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._auto_encoder = auto_encoder
        self._rsr_dim = rsr_dim

    def _reconstruct_loss(self, x: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x_pred = x_pred.to(self.device)

        return (x - x_pred).norm(dim=1).sum()

    def _pca_loss(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        A = self._auto_encoder.rsr.A.to(self.device)
        return (z - z.matmul(A).matmul(A.T)).norm(dim=1).sum()

    def _project_loss(self) -> torch.Tensor:
        A = self._auto_encoder.rsr.A.to(self.device)
        E = torch.eye(self._rsr_dim).to(self.device)
        return torch.norm(A.T.matmul(A) - E)

    def _load_rsr_optimizer(self) -> Optimizer:
        return Adam(self._auto_encoder.rsr.parameters(), lr=10 * self._learning_rate)

    def train(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        # Optimizer
        optimizer = self._load_optimizer()
        rsr_optimizer = self._load_rsr_optimizer()

        self._auto_encoder.train()

        z = self._auto_encoder.encode(x)
        x_pred: torch.Tensor = self._auto_encoder(x)

        # Compute loss
        loss = (
            self._reconstruct_loss(x, x_pred) + self._lambda1 * self._pca_loss(z) + self._lambda2 * self._project_loss()
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        rsr_optimizer.step()
        rsr_optimizer.zero_grad()

        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x_pred: torch.Tensor = self._auto_encoder(x)
        cos = nn.CosineSimilarity()

        return (cos(x, x_pred) + 1) / 2
