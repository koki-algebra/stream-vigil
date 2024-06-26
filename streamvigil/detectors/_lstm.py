import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        _, (hn, cn) = self.lstm(x, (h0, c0))

        return hn, cn


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hn: int, cn: int):
        out, _ = self.lstm(x, (hn, cn))
        out = self.fc(out)

        return out


class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(
            input_size,
            hidden_size,
            num_layers,
        )
        self.decoder = LSTMDecoder(
            input_size,
            hidden_size,
            num_layers,
            input_size,
        )

    def forward(self, x: torch.Tensor):
        hn, cn = self.encoder(x)
        decoder_input = torch.zeros(x.size(0), x.size(1), x.size(2)).to(x.device)
        out = self.decoder(decoder_input, hn, cn)

        return out
