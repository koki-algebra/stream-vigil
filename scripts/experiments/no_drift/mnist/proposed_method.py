from logging import getLogger
from logging.config import dictConfig

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil.core import Model, ModelPool
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import set_seed

RANDOM_STATE = 80
TRAIN_BATCH_SIZE = 128
INIT_BATCHES = 20


def main():
    set_seed(RANDOM_STATE)

    with open("./notebooks/logging.yml", encoding="utf-8") as file:
        config = safe_load(file)
    dictConfig(config)
    logger = getLogger(__name__)

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
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    auto_encoder = BasicAutoEncoder(
        encoder_dims=[784, 588, 392, 196],
        decoder_dims=[196, 392, 588, 784],
        batch_norm=True,
    )
    detector = BasicDetector(auto_encoder)

    # Model Pool
    model_pool = ModelPool[Model](
        detector,
        historical_window_size=500,
        latest_window_size=500,
        last_trained_size=500,
        window_gap=500,
    )

    # Number of false positives
    fp_cnt = 0

    losses = []
    detected = []

    # Training
    for X, _ in train_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_window(X)

        current_model = model_pool.get_model(model_pool.current_model_id)

        if current_model.num_batches > INIT_BATCHES:
            # Concept Drift detection
            if current_model.is_drift():
                logger.info("concept drift detected!")

                detected.append(1)

                fp_cnt += 1

                adapted_model_id = model_pool.find_adapted_model()
                if adapted_model_id is not None:
                    model_pool.current_model_id = adapted_model_id

                    logger.info(f"find adapted model: {adapted_model_id}")
                else:
                    # Add new model
                    new_model_id = model_pool.add_model()
                    model_pool.current_model_id = new_model_id

                    logger.info(f"add new model: {new_model_id}")
            else:
                detected.append(0)

        # Train current model
        loss = model_pool.stream_train(X)
        losses.append(loss.detach())

    logger.info(f"Number of false positives: {fp_cnt}")

    losses = np.array(losses)
    detected = np.array(detected)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot losses
    color = "tab:blue"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(range(len(losses)), losses, color=color, alpha=0.7)
    ax1.tick_params(axis="y", labelcolor=color)

    # Create a twin Axes for detected events
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Drift Detected", color=color)

    # Plot detected events
    detected_indices = np.where(detected == 1)[0]
    ax2.scatter(detected_indices, [1] * len(detected_indices), color=color, marker="o", s=50)

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["", "Drift Detected"])
    ax2.tick_params(axis="y", labelcolor=color)

    # Set title and adjust layout
    plt.title("Loss and Drift Detection over Iterations (MNIST)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
