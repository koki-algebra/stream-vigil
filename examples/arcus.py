from logging import getLogger
from logging.config import dictConfig

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from yaml import safe_load

from streamvigil import ARCUS, CustomDataset
from streamvigil.core import AutoEncoder
from streamvigil.detectors import BasicDetector


class BasicAutoEncoder(AutoEncoder):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def main():
    # Logger
    with open("./examples/logging.yml", encoding="utf-8") as file:
        config = safe_load(file)
    dictConfig(config)

    logger = getLogger(__name__)

    # Create ARCUS instance
    auto_encoder = BasicAutoEncoder(input_dim=128, hidden_dim=96, latent_dim=64)
    detector = BasicDetector(auto_encoder)
    arcus = ARCUS(detector)

    # Dataset
    dataset = CustomDataset(csv_file="./data/GAS.csv")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    # Initialize model pool with first batch
    logger.info("Start initializing model pool...")
    init_features, _ = next(iter(dataloader))
    arcus.init(x=init_features)
    logger.info("Complete model pool initialization!")

    all_scores = []
    all_labels = []

    # Train the models
    logger.info("Start anomaly detection simulation...")
    for x, y in dataloader:
        scores = arcus.run(x)
        all_scores.append(scores.detach().cpu().numpy())
        all_labels.append(y.numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # Compute ROC AUC score
    roc_auc = roc_auc_score(all_labels, all_scores)
    logger.info("ROC AUC Score: {}".format(roc_auc))

    logger.info("Completed anomaly detection simulation!")


if __name__ == "__main__":
    main()
