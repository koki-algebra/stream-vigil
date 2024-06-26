import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC

from streamvigil import CSVDataset
from streamvigil.detectors import BasicAutoEncoder
from streamvigil.utils import set_seed

learning_rate = 1e-3
train_batch_size = 128
test_batch_size = 64
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    random_state = 80
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

    # Model
    auto_encoder = BasicAutoEncoder(
        encoder_dims=[196, 147, 98, 49],
        decoder_dims=[49, 98, 147, 196],
        batch_norm=True,
    )
    auto_encoder.to(device)

    # Loss function
    criterion = nn.MSELoss(reduction="none")

    # Optimizer
    optimizer = Adam(
        auto_encoder.parameters(),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        train_loop(
            auto_encoder,
            train_loader,
            criterion,
            optimizer,
        )

    # Evaluation
    test_loop(
        auto_encoder,
        test_loader,
    )


def train_loop(model: nn.Module, loader: DataLoader, criterion, optimizer: Optimizer):
    size = len(loader.dataset)

    model.train()
    for batch, (X, y) in enumerate(loader):
        X = X.to(device)

        X_pred = model(X)
        losses: torch.Tensor = criterion(X_pred, X).mean(dim=1)

        normal_losses = losses[y == 0]
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

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * train_batch_size + len(X)

            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(model: nn.Module, loader: DataLoader):
    model.eval()

    # Area Under the ROC Curve
    auroc = BinaryAUROC()
    # Area Under the Precision-Recall Curve
    auprc = BinaryAUPRC()

    with torch.no_grad():
        normal_scores = []
        anomaly_scores = []
        for X, y in loader:
            X = X.to(device)

            X_pred = model(X)
            scores = anomaly_score(X_pred, X)
            auroc.update(scores, y)
            auprc.update(scores, y)

            normal_scores.extend(scores[y == 0].tolist())
            anomaly_scores.extend(scores[y == 1].tolist())

    print(f"AUROC Score: {auroc.compute():0.3f}")
    print(f"AUPRC Score: {auprc.compute():0.3f}")

    print(f"Average score for normal data: {torch.tensor(normal_scores).mean().item():0.5f}")
    print(f"Average score for anomaly data: {torch.tensor(anomaly_scores).mean().item():0.5f}")


def anomaly_score(X_pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (X_pred - X).norm(dim=1).square()


if __name__ == "__main__":
    main()
