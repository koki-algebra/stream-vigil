import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC

from streamvigil import CSVDataset
from streamvigil.detectors import LSTMAutoEncoder
from streamvigil.utils import set_seed


def anomaly_score(X_pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (X_pred - X).norm(dim=1).square()


def main():
    random_state = 80
    set_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = 29
    hidden_size = 15
    learning_rate = 1e-3
    train_batch_size = 128
    test_batch_size = 64

    data_path = "./data/ADBench/13_fraud.csv.gz"

    train_dataset = CSVDataset(
        data_path,
        train=True,
        test_size=0.2,
        shuffle=True,
        random_state=random_state,
    )

    test_dataset = CSVDataset(
        data_path,
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
        hidden_size=hidden_size,
        num_layers=5,
    )

    auto_encoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(auto_encoder.parameters(), lr=learning_rate)

    # Training
    epochs = 5
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
