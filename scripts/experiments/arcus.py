from logging import getLogger
from logging.config import dictConfig

from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil import ARCUS
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import set_seed

train_batch_size = 128
test_batch_size = 64
epochs = 5


def main():
    random_state = 80
    set_seed(random_state)

    with open("./notebooks/logging.yml", encoding="utf-8") as file:
        config = safe_load(file)
    dictConfig(config)
    logger = getLogger(__name__)

    auto_encoder = BasicAutoEncoder(
        encoder_dims=[784, 588, 392, 196],
        decoder_dims=[196, 392, 588, 784],
        batch_norm=True,
    )
    detector = BasicDetector(auto_encoder)

    arcus = ARCUS(detector, logger)
    arcus.init()

    # Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data/pytorch",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
    )

    # Training
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for batch, (X, _) in enumerate(train_loader):
            X = X.view(X.size(0), -1)

            if batch % 100 == 0:
                arcus.stream_train(X, is_logging=True)
            else:
                arcus.stream_train(X)

    # Test
    test_dataset = datasets.MNIST(
        root="./data/pytorch",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
    )

    # Area Under the ROC Curve
    auroc = BinaryAUROC()
    # Area Under the Precision-Recall Curve
    auprc = BinaryAUPRC()

    for X, y in test_loader:
        X = X.view(X.size(0), -1)

        scores = arcus.predict(X)
        auroc.update(scores, y)
        auprc.update(scores, y)

    logger.info(f"AUROC Score: {auroc.compute():0.3f}")
    logger.info(f"AUPRC Score: {auprc.compute():0.3f}")


if __name__ == "__main__":
    main()
