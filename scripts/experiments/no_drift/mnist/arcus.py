from logging import getLogger
from logging.config import dictConfig

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil._arcus_model import ARCUSModel
from streamvigil._arcus_model_pool import ARCUSModelPool
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import set_seed

train_batch_size = 128
test_batch_size = 64

loss_color = "#00ADD8"
reliability_color = "#00A29C"
detected_color = "#CE3262"


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

    # Train initial model
    init_model_id = model_pool.add_model()

    logger.info("start training the initial model")
    for batch, (X, _) in enumerate(train_loader):
        X = X.view(X.size(0), -1)
        model_pool.stream_train(init_model_id, X)
    logger.info("finish training initial model")

    reliabilities = []
    losses = []
    detected = []

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

            detected.append(1)
        else:
            model_id = model_pool.find_most_reliable_model()
            loss = model_pool.stream_train(model_id, X)

            detected.append(0)

        if model_id == init_model_id:
            model = model_pool.get_model(model_id)
            reliabilities.append(model.reliability)
            losses.append(loss.detach())

    losses = np.array(losses)
    reliabilities = np.array(reliabilities)
    detected = np.array(detected[: len(losses) - 1])

    # Plot 1: Losses and Reliabilities
    plt.figure(figsize=(12, 6))
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(losses, color=loss_color, label="Losses")
    ax1.set_ylabel("Loss", color=loss_color)
    ax1.tick_params(axis="y", labelcolor=loss_color)
    ax1.set_ylim(0.0, 0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(reliabilities, color=reliability_color, label="Reliabilities")
    ax1_twin.set_ylabel("Reliability", color=reliability_color)
    ax1_twin.tick_params(axis="y", labelcolor=reliability_color)
    ax1_twin.set_ylim(0.0, 1.2)

    ax1.set_title("Losses and Reliabilities (MNIST)")
    ax1.set_xlabel("Iterations")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig("losses_and_reliabilities.png")
    plt.close(fig1)

    # Plot 2: Losses and Detected
    plt.figure(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    ax2.plot(losses, color=loss_color, label="Losses")
    ax2.set_ylabel("Loss", color=loss_color)
    ax2.tick_params(axis="y", labelcolor=loss_color)
    ax2.set_ylim(0.0, 0.5)

    ax2_twin = ax2.twinx()
    detected_indices = np.where(detected == 1)[0]
    ax2_twin.scatter(detected_indices, [1] * len(detected_indices), color=detected_color, label="Detected", alpha=0.6)
    ax2_twin.set_ylabel("Detected", color=detected_color)
    ax2_twin.tick_params(axis="y", labelcolor=detected_color)
    ax2_twin.set_yticks([0, 1])
    ax2_twin.set_ylim(-0.1, 1.1)

    ax2.set_title("Losses and Detected Drift (MNIST)")
    ax2.set_xlabel("Iterations")

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig("losses_and_detected.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
