from logging import getLogger
from logging.config import dictConfig

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil._arcus_model import ARCUSModel
from streamvigil._arcus_model_pool import ARCUSModelPool
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import set_seed

train_batch_size = 128
test_batch_size = 64


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

    model_pool = ARCUSModelPool[ARCUSModel](detector)

    # Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(
        root="./data/pytorch",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
    )

    # Train initial model
    init_model_id = model_pool.add_model()

    logger.info("start training the initial model")
    for batch, (X, _) in enumerate(train_loader):
        X = X.view(X.size(0), -1)
        model_pool.stream_train(init_model_id, X)
    logger.info("finish training initial model")

    reliabilities = []
    losses = []

    # Training
    for batch, (X, _) in enumerate(train_loader):
        X = X.view(X.size(0), -1)

        model_pool.update_reliability(X)

        if model_pool.is_drift():
            logger.info("concept drift detected!")
            model_id = model_pool.add_model()

            logger.info(f"new model {model_id} is added")

            loss = model_pool.stream_train(model_id, X)

            if model_pool.compress(X, model_id):
                logger.info("model pool compressed!")
        else:
            model_id = model_pool.find_most_reliable_model()
            loss = model_pool.stream_train(model_id, X)

        if init_model_id == model_id:
            model = model_pool.get_model(model_id)
            reliabilities.append(model.reliability)
            losses.append(loss.detach())

    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()
    ax1.plot(losses, marker="o", linestyle="-", color="#00ADD8", markersize=3, label="Loss")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Loss", color="#00ADD8")
    ax1.tick_params(axis="y", labelcolor="#00ADD8")
    ax1.set_ylim(0.0, 0.3)

    ax2 = ax1.twinx()
    ax2.plot(reliabilities, marker="s", linestyle="-", color="#00A29C", markersize=3, label="Model Reliability")
    ax2.set_ylabel("Model Reliability", color="#00A29C")
    ax2.tick_params(axis="y", labelcolor="#00A29C")
    ax2.set_ylim(0.0, 1.2)

    plt.title("Changes in Training Loss and Reliability (Fashion-MNIST)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
