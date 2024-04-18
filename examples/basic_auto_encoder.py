from logging import getLogger
from logging.config import dictConfig

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
    auto_encoder = BasicAutoEncoder(
        encoder_dims=[500, 450, 400, 350, 300, 250],
        decoder_dims=[250, 300, 350, 400, 450, 500],
        batch_norm=True,
    )
    detector = BasicDetector(auto_encoder)

    random_state = 42

    # Load dataset
    train_data = ADBenchDataset(
        "./data/9_census.npz",
        train=True,
        random_state=random_state,
    )
    test_data = ADBenchDataset(
        "./data/9_census.npz",
        train=False,
        random_state=random_state,
    )

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=512)
    test_loader = DataLoader(test_data, batch_size=256)

    # Training
    epochs = 10
    logger.info("Start training the model...")
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch+1}")
        for batch, (X, _) in enumerate(train_loader):
            loss = detector.train(X)

            if batch % 100 == 0:
                logger.info(f"Loss: {loss.item():>7f}")
    logger.info("Completed training the model!")

    # Evaluation
    metrics = BinaryAUROC()
    logger.info("Start evaluating the model...")

    for X, y in test_loader:
        scores = detector.predict(X)
        metrics.update(scores, y)

    # Compute AUROC score   
    logger.info(f"AUROC Score: {metrics.compute()}")

    logger.info("Completed the evaluation of the model!")


if __name__ == "__main__":
    main()
