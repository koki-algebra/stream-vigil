from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torchvision import datasets, transforms

from streamvigil.detectors import BasicAutoEncoder
from streamvigil.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_by_label(dataset: Dataset, labels: List):
    indices = [i for i, target in enumerate(dataset.targets) if target in labels]
    dataset.data = dataset.data[indices]
    dataset.targets = dataset.targets[indices]
    return dataset


def anomaly_score(X_pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (X_pred - X).norm(dim=1).square()


def main():
    random_state = 80
    set_seed(random_state)

    transform = transforms.Compose([transforms.ToTensor()])
    learning_rate = 1e-3
    train_batch_size = 128
    test_batch_size = 64
    epochs = 10

    train_dataset = datasets.MNIST(
        root="./data/MNIST",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="./data/MNIST",
        train=False,
        download=True,
        transform=transform,
    )

    # filtering
    train_dataset = filter_by_label(train_dataset, [1, 2, 3])
    test_dataset = filter_by_label(test_dataset, [1, 2, 3, 9])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
    )

    auto_encoder = BasicAutoEncoder(
        encoder_dims=[784, 588, 392, 196],
        decoder_dims=[196, 392, 588, 784],
        batch_norm=True,
    )
    auto_encoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        auto_encoder.parameters(),
        lr=learning_rate,
    )

    # Training
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        for batch, (X, _) in enumerate(train_loader):
            X: torch.Tensor = X.to(device)
            X = X.view(X.size(0), -1)
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
        for X, y in test_loader:
            X: torch.Tensor = X.to(device)
            y: torch.Tensor = y.to(device)

            X = X.view(X.size(0), -1)
            X_pred = auto_encoder(X)

            scores = anomaly_score(X_pred, X)

            y[y != 9] = 0
            y[y == 9] = 1

            auroc.update(scores, y)
            auprc.update(scores, y)

    print(f"AUROC Score: {auroc.compute():0.3f}")
    print(f"AUPRC Score: {auprc.compute():0.3f}")


if __name__ == "__main__":
    main()
