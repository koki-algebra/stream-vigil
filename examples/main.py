from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset


class NPYDataset(Dataset):
    def __init__(self, filepath: str, train=True, test_size=0.3, shuffle=False) -> None:
        super().__init__()

        data = np.load(filepath)
        X = data["X"]
        y = data["y"]
        permutation = np.random.permutation(len(X))
        X = X[permutation]
        y = y[permutation]

        split_idx = int(len(X) * test_size)

        X_train, X_test = X[split_idx:], X[:split_idx]
        y_train, y_test = y[split_idx:], y[:split_idx]

        if train:
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train)
        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class AutoEncoder(nn.Module):
    def __init__(self, encoder_dims: List[int], decoder_dims: List[int]) -> None:
        super().__init__()

        self.encoder = nn.Sequential()
        for input_dim, output_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            self.encoder.append(nn.Linear(input_dim, output_dim))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm1d(output_dim))

        self.decoder = nn.Sequential()
        for input_dim, output_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            self.decoder.append(nn.Linear(input_dim, output_dim))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.BatchNorm1d(output_dim))

    def forward(self, X: torch.Tensor):
        return self.decoder(self.encoder(X))


def train_loop(loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer):
    model.train()
    for batch, (X, _) in enumerate(loader):
        X_pred = model(X)

        loss: torch.Tensor = criterion(X, X_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"loss = {loss}")


def test_loop(loader: DataLoader, model: nn.Module, criterion: nn.Module):
    model.eval()
    loss = 0

    num_batches = len(loader)

    with torch.no_grad():
        for X, y in loader:
            X_pred = model(X)
            loss += criterion(X, X_pred)

    loss /= num_batches

    print(f"Avg Loss: {loss}")


def main():
    train_dataset = NPYDataset("./data/11_donors.npz", train=True)
    test_dataset = NPYDataset("./data/11_donors.npz", train=True)
    train_loader = DataLoader(train_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = AutoEncoder(encoder_dims=[10, 5, 2], decoder_dims=[2, 5, 10])

    criterion = nn.MSELoss()

    optimizer = Adam(model.parameters())
    epochs = 16

    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        train_loop(train_loader, model, criterion, optimizer)

    test_loop(test_loader, model, criterion)


if __name__ == "__main__":
    main()
