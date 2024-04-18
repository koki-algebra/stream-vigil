from logging import getLogger
from logging.config import dictConfig

from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC
from yaml import safe_load

from streamvigil import ARCUS, ADBenchDataset
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

    arcus = ARCUS(detector, max_epochs=20)

    random_state = 42

    # Load dataset
    data = ADBenchDataset(
        "./data/9_census.npz",
        train=True,
        test_size=0.01,
        random_state=random_state,
    )

    # DataLoader
    loader = DataLoader(data, batch_size=1024)

    # Initialize model pool
    logger.info("Start initializing model pool...")
    X, _ = next(iter(loader))
    arcus.init(X)
    logger.info("Completed initializing model pool!")

    # ARCUS simulation
    metrics = BinaryAUROC()

    logger.info("Start ARCUS simulation...")
    for X, y in loader:
        scores = arcus.run(X)
        metrics.update(scores, y)

    # Compute AUROC score
    logger.info(f"AUROC Score: {metrics.compute()}")

    logger.info("Completed ARCUS simulation!")


if __name__ == "__main__":
    main()
