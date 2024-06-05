from logging import getLogger
from logging.config import dictConfig

from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryPrecisionRecallCurve
from yaml import safe_load

from streamvigil import ARCUS, CSVDataset
from streamvigil.detectors import RAPP


def main():
    # Logger
    with open("./examples/logging.yml", encoding="utf-8") as file:
        config = safe_load(file)
    dictConfig(config)
    logger = getLogger(__name__)

    random_state = 42

    # Dataset
    data = CSVDataset(
        "./data/INSECTS/INSECTS_Abr.csv.gz",
        train=True,
        test_size=0.0,
        random_state=random_state,
    )

    # DataLoader
    loader = DataLoader(
        data,
        batch_size=128,
    )

    detector = RAPP(
        encoder_dims=[33, 28, 23, 18],
        decoder_dims=[18, 23, 28, 33],
        batch_norm=True,
    )

    arcus = ARCUS(detector, max_epochs=5)

    # Initialize model pool
    logger.info("Start initializing model pool...")
    X, _ = next(iter(loader))
    arcus.init(X)
    logger.info("Completed initializing model pool!")

    # ARCUS simulation
    auroc = BinaryAUROC()
    auprc = BinaryAUPRC()
    pr_curve = BinaryPrecisionRecallCurve()

    logger.info("Start ARCUS simulation...")
    for X, y in loader:
        scores = arcus.run(X)
        auroc.update(scores, y)
        auprc.update(scores, y)
        pr_curve.update(scores, y)

    # Compute AUROC score
    logger.info(f"AUROC Score: {auroc.compute()}")
    logger.info(f"AUPRC Score: {auprc.compute()}")

    logger.info("Completed ARCUS simulation!")


if __name__ == "__main__":
    main()
