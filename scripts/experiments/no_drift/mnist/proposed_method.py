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

    ax1.plot(losses, color="#00ADD8", label="Losses")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss", color="#00ADD8")
    ax1.tick_params(axis="y", labelcolor="#00ADD8")

    # Create a twin axis for the detected scatter plot
    ax2 = ax1.twinx()

    detected_indices = np.where(detected == 1)[0]
    ax2.scatter(detected_indices, [1] * len(detected_indices), color="#CE3262", label="Drift Detected", alpha=0.6)
    ax2.set_ylabel("Drift Detected", color="#CE3262")
    ax2.tick_params(axis="y", labelcolor="#CE3262")
    ax2.set_yticks([0, 1])
    ax2.set_ylim(-0.1, 1.1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Losses and Drift Detected over Iterations (MNIST)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
