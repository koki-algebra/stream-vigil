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
        return x.matmul(self.A.to(x.device))


class _RSRAE(AutoEncoder):
    """
    Robust Subspace Recovery Auto Encoder (RSRAE)
    """

    def __init__(self, encoder_dims: List[int], decoder_dims: List[int], rsr_dim: int, batch_norm=False) -> None:
        super().__init__()

        # Encoder
        self.encoder = self._build_network(encoder_dims, batch_norm)

        # Robust Subspace Recovery layer (RSR layer)
        self.rsr = RSR(encoder_dims[-1], rsr_dim)

        # Decoder
        self.decoder = self._build_network(decoder_dims, batch_norm)

        # Register forward hook
        self._latent_value: torch.Tensor | None = None

        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self._latent_value = output.detach()

        self.encoder[-1].register_forward_hook(hook)

    def _build_network(self, dims: List[int], batch_norm=False):
        network = nn.Sequential()
        for i, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            network.append(nn.Linear(input_dim, output_dim))
            if i != len(dims) - 2:
                if batch_norm:
                    network.append(nn.BatchNorm1d(output_dim))
                network.append(nn.ReLU())

        return network

    def get_latent(self) -> torch.Tensor | None:
        return self._latent_value

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
        batch_norm=False,
        lambda1=1.0,
        lambda2=1.0,
    ) -> None:
        auto_encoder = _RSRAE(
            encoder_dims,
            decoder_dims,
            rsr_dim,
            batch_norm=batch_norm,
        )
        super().__init__(auto_encoder)
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._auto_encoder = auto_encoder
        self._rsr_dim = rsr_dim

    def _reconstruct_loss(self, x: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        return (x - x_pred).norm(p=1, dim=1).mean()

    def _pca_loss(self, z: torch.Tensor) -> torch.Tensor:
        A = self._auto_encoder.rsr.A
        z = z.to(A.device)
        return (z - z.matmul(A).matmul(A.T)).norm(p=1, dim=1).mean()

    def _project_loss(self) -> torch.Tensor:
        A = self._auto_encoder.rsr.A
        E = torch.eye(self._rsr_dim, device=A.device)
        return (A.T.matmul(A) - E).norm().square()

    def _load_rsr_optimizer(self) -> Optimizer:
        return Adam(self._auto_encoder.rsr.parameters(), lr=10 * self._learning_rate)

    def train(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        # Optimizer
        optimizer = self._load_optimizer()
        rsr_optimizer = self._load_rsr_optimizer()

        self._auto_encoder.train()

        x_pred: torch.Tensor = self._auto_encoder(x)
        z = self._auto_encoder.get_latent()

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
