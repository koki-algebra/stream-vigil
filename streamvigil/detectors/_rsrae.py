from typing import List, override

import torch.nn as nn
from torch import Tensor

from streamvigil.core import AutoEncoder


class RSRAE(AutoEncoder):
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
        self.rsr = nn.Linear(encoder_dims[-1], rsr_dim, bias=False)

        # Decoder
        input_dim = rsr_dim
        self.decoder = nn.Sequential()
        for i, output_dim in enumerate(decoder_dims):
            self.decoder.append(nn.Linear(input_dim, output_dim))
            if i != len(decoder_dims) - 1:
                self.decoder.append(nn.ReLU())
                self.decoder.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    @override
    def forward(self, x: Tensor) -> Tensor:
        z = self.rsr(self.encode(x))
        return self.decode(z)
