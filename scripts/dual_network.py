from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC

from streamvigil import CSVDataset
from streamvigil.utils import set_seed

learning_rate = 1e-3
train_batch_size = 128
test_batch_size = 64
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_pred = self.decoder(z)

        return z, x_pred


class Classifier(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        self.cls = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.cls(z).squeeze()


def main():
    random_state = 84
    set_seed(random_state)

    # Dataset
    train_data = CSVDataset(
        "./data/ADBench/3_backdoor.csv.gz",
        train=True,
        test_size=0.2,
        random_state=random_state,
    )
    test_data = CSVDataset(
        "./data/ADBench/3_backdoor.csv.gz",
        train=False,
        test_size=0.2,
        random_state=random_state,
    )

    # DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
    )

    # AutoEncoder
    auto_encoder = AutoEncoder(
        encoder_dims=[196, 147, 98, 49],
        decoder_dims=[49, 98, 147, 196],
        batch_norm=True,
    )
    auto_encoder.to(device)

    # Classifier
    cls = Classifier(latent_dim=49)
    cls.to(device)

    # Loss function
    criterion_autoencoder = nn.MSELoss()
    criterion_classifier = nn.BCELoss()

    # Optimizer
    optimizer_autoencoder = optim.Adam(auto_encoder.parameters(), lr=learning_rate)
    optimizer_classifier = optim.Adam(cls.parameters(), lr=learning_rate)

    # Train AutoEncoder
    print("start train the AutoEncoder")
    auto_encoder.train()
    for epoch in range(epochs):
        for X, _ in train_loader:
            X = X.to(device)
            _, X_pred = auto_encoder(X)
            loss = criterion_autoencoder(X_pred, X)

            optimizer_autoencoder.zero_grad()
            loss.backward()
            optimizer_autoencoder.step()

        print(f"Epoch [{epoch+1}/{epochs}], loss: {loss.item():.4f}")
    print("training is completed")

    # Train Classifier
    print("start train the Classifier")
    cls.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            Z, _ = auto_encoder(X)
            y_pred = cls(Z)

            loss = criterion_classifier(y_pred, y)

            optimizer_classifier.zero_grad()
            loss.backward()
            optimizer_classifier.step()

        print(f"Epoch [{epoch+1}/{epochs}], loss: {loss.item():.4f}")
    print("training is completed")

    # Evaluation
    cls.eval()

    # Area Under the ROC Curve
    auroc = BinaryAUROC()
    # Area Under the Precision-Recall Curve
    auprc = BinaryAUPRC()

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)

            Z, _ = auto_encoder(X)
            y_pred = cls(Z)

            auroc.update(y_pred, y)
            auprc.update(y_pred, y)

    print(f"AUROC Score: {auroc.compute():0.3f}")
    print(f"AUPRC Score: {auprc.compute():0.3f}")


if __name__ == "__main__":
    main()
