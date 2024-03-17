from logging import getLogger

import torch
from torch.nn import MSELoss

from streamvigil.core import AnomalyDetector, AutoEncoder


class BasicDetector(AnomalyDetector):
    def __init__(self, auto_encoder: AutoEncoder, max_train_epochs=100) -> None:
        super().__init__(auto_encoder)
        self._max_train_epochs = max_train_epochs

        # Loss function
        self._loss_fn = MSELoss()

    def train(self, x: torch.Tensor) -> None:
        logger = getLogger(__name__)

        # Optimizer
        optimizer = self._load_optimizer()

        # Training the model
        self._auto_encoder.train()
        for epoch in range(self._max_train_epochs):
            logger.info("Epoch {}".format(epoch))

            # Compute prediction and loss
            x_pred: torch.Tensor = self._auto_encoder(x)
            loss: torch.Tensor = self._loss_fn(x_pred, x)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x_pred: torch.Tensor = self._auto_encoder(x)

        # Square error
        errs = (x - x_pred).pow(2).sum(dim=1)

        # Anomaly scores
        scores = errs.sigmoid()

        return scores
