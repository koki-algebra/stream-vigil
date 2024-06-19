from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import BinaryAUPRC, BinaryAUROC

from streamvigil.utils import set_seed


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


class CSVDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        train=True,
        test_size=0.3,
        shuffle=False,
        random_state: float | None = None,
    ) -> None:
        super().__init__()

        data = pd.read_csv(filepath)
        X_train = data.iloc[:, :-1].to_numpy()
        y_train = data.iloc[:, -1].to_numpy()

        if test_size != 0.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train,
                y_train,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state,
            )

        if train:
            X_train = torch.from_numpy(X_train.astype(np.float32))
            y_train = torch.from_numpy(y_train.astype(np.float32))
            idx = (y_train == 0).nonzero(as_tuple=False).squeeze()
            self.X = X_train[idx]
            self.y = y_train[idx]
        else:
            self.X = torch.from_numpy(X_test.astype(np.float32))
            self.y = torch.from_numpy(y_test.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def anomaly_score(X_pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (X_pred - X).norm(dim=1).square()


def main():
    random_state = 80
    set_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = 27
    learning_rate = 1e-3
    train_batch_size = 128
    test_batch_size = 64

    train_dataset = CSVDataset(
        "./data/ADBench/1_ALOI.csv.gz",
        train=True,
        test_size=0.2,
        shuffle=True,
        random_state=random_state,
    )

    test_dataset = CSVDataset(
        "./data/ADBench/1_ALOI.csv.gz",
        train=False,
        test_size=0.2,
        shuffle=True,
        random_state=random_state,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
    )

    auto_encoder = LSTMAutoEncoder(
        input_size=n_features,
        hidden_size=54,
        num_layers=5,
    )

    auto_encoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(auto_encoder.parameters(), lr=learning_rate)

    # Training
    epochs = 10
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        for batch, (X, _) in enumerate(train_loader):
            X: torch.Tensor = X.unsqueeze(1).to(device)
            X_pred = auto_encoder(X)

            loss = criterion(X_pred, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss = loss.item()
                current = batch * train_batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_dataset):>5d}]")

    # Evaluation
    auto_encoder.eval()

    # Area Under the ROC Curve
    auroc = BinaryAUROC()
    # Area Under the Precision-Recall Curve
    auprc = BinaryAUPRC()

    with torch.no_grad():
        normal_scores = []
        anomaly_scores = []
        for X, y in test_loader:
            X: torch.Tensor = X.unsqueeze(1).to(device)
            y: torch.Tensor = y.to(device)

            X_pred: torch.Tensor = auto_encoder(X)

            X = X.squeeze(1)
            X_pred = X_pred.squeeze(1)
            scores = anomaly_score(X_pred, X)

            normal_scores.extend(scores[y == 0].tolist())
            anomaly_scores.extend(scores[y == 1].tolist())

            auroc.update(scores, y)
            auprc.update(scores, y)

    print(f"AUROC Score: {auroc.compute():0.3f}")
    print(f"AUPRC Score: {auprc.compute():0.3f}")

    print(f"Average score for normal data: {torch.tensor(normal_scores).mean().item():0.5f}")
    print(f"Average score for anomaly data: {torch.tensor(anomaly_scores).mean().item():0.5f}")


if __name__ == "__main__":
    main()
