from logging import getLogger
from logging.config import dictConfig

import torch
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC
from yaml import safe_load

from streamvigil import ADBenchDataset
from streamvigil.detectors import BasicAutoEncoder, BasicDetector


def main():
    # Logger
    with open("./examples/logging.yml", encoding="utf-8") as file:
        config = safe_load(file)
    dictConfig(config)
    logger = getLogger(__name__)

    # Anomaly Detector
    auto_encoder = BasicAutoEncoder(encoder_dims=[10, 8, 5], decoder_dims=[5, 8, 10], batch_norm=True)
    detector = BasicDetector(auto_encoder)

    random_state = 42

    # Load dataset
    train_data = ADBenchDataset("./data/11_donors.npz", train=True, random_state=random_state)
    test_data = ADBenchDataset("./data/11_donors.npz", train=False, random_state=random_state)

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=128)
    test_loader = DataLoader(test_data, batch_size=64)

    # Training
    epochs = 5
    logger.info("Start training the model...")
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch+1}")
        for batch, (X, _) in enumerate(train_loader):
            detector.train(X)
    logger.info("Completed training the model!")

    # Evaluation
    logger.info("Start evaluating the model...")

    all_scores = []
    all_labels = []

    for X, y in test_loader:
        scores = detector.predict(X)
        all_scores.append(scores)
        all_labels.append(y)

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute AUROC score
    metrics = BinaryAUROC()
    metrics.update(all_scores, all_labels)
    logger.info(f"AUROC Score: {metrics.compute()}")

    logger.info("Completed the evaluation of the model!")


if __name__ == "__main__":
    main()
