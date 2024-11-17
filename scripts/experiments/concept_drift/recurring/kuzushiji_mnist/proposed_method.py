from logging import getLogger
from logging.config import dictConfig
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil.core import Model, ModelPool
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import filter_index, set_seed, to_anomaly_labels

RANDOM_STATE = 80
TEST_BATCH_SIZE = 128
LOSS_COLOR = "#00ADD8"
DETECTED_COLOR = "#CE3262"

TRAIN_BATCH_SIZE = 128
INIT_BATCHES = 70
ALPHA = 0.05

LATEST_WINDOW_SIZE = 500
HISTORICAL_WINDOW_SIZE = 500
LAST_TRAINED_WINDOW_SIZE = 500
WINDOW_GAP = 500


def get_data_loader(
    normal_labels: List[int],
    anomaly_labels: List[int],
    anomaly_ratio=0.01,
    train=True,
):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.KMNIST(
        root="./data/pytorch",
        train=train,
        download=True,
        transform=transform,
    )

    filtered_idx = filter_index(
        dataset.targets,
        normal_labels=normal_labels,
        anomaly_labels=anomaly_labels,
        anomaly_ratio=anomaly_ratio,
    )

    dataset.targets = to_anomaly_labels(
        dataset.targets[filtered_idx],
        normal_labels=normal_labels,
    )
    dataset.data = dataset.data[filtered_idx]

    loader = DataLoader(
        dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
    )

    return loader


def main():
    set_seed(RANDOM_STATE)

    with open("./notebooks/logging.yml", encoding="utf-8") as file:
        config = safe_load(file)
    dictConfig(config)
    logger = getLogger(__name__)

    # Model
    auto_encoder = BasicAutoEncoder(
        encoder_dims=[784, 588, 392, 196],
        decoder_dims=[196, 392, 588, 784],
        batch_norm=True,
    )
    detector = BasicDetector(auto_encoder)

    # Model Pool
    model_pool = ModelPool[Model](
        detector,
        historical_window_size=HISTORICAL_WINDOW_SIZE,
        latest_window_size=LATEST_WINDOW_SIZE,
        last_trained_size=LAST_TRAINED_WINDOW_SIZE,
        window_gap=WINDOW_GAP,
        alpha=ALPHA,
    )

    # Data loader
    train_a_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[7, 8, 9],
        anomaly_ratio=0.001,
    )
    train_b_loader = get_data_loader(
        normal_labels=[3, 4],
        anomaly_labels=[7, 8, 9],
        anomaly_ratio=0.001,
    )
    train_c_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[7, 8, 9],
        anomaly_ratio=0.001,
    )
    test_a_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[0, 3, 4, 5, 6, 7, 8, 9],
        train=False,
    )
    test_b_loader = get_data_loader(
        normal_labels=[3, 4],
        anomaly_labels=[0, 5, 6, 7, 8, 9],
        train=False,
    )
    test_c_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[0, 5, 6, 7, 8, 9],
        train=False,
    )

    auroc = BinaryAUROC()
    auprc = BinaryAUPRC()

    losses = []
    detected = []

    # Training for concept A
    for X, y in train_a_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_window(X)

        current_model = model_pool.get_model(model_pool.current_model_id)

        if current_model.num_batches > INIT_BATCHES:
            # Concept Drift detection
            if current_model.is_drift():
                logger.info("concept drift detected!")

                detected.append(1)

                adapted_model_id = model_pool.find_adapted_model()
                if adapted_model_id is not None:
                    model_pool.current_model_id = adapted_model_id

                    logger.info(f"find adapted model: {adapted_model_id}")
                else:
                    # Add new model
                    new_model_id = model_pool.add_model()
                    model_pool.current_model_id = new_model_id

                    logger.info(f"add new model: {new_model_id}")

        if not (current_model.num_batches > INIT_BATCHES and current_model.is_drift()):
            detected.append(0)

        # Train current model
        loss = model_pool.stream_train(X)

        losses.append(loss.detach())

    # Evaluation for concept A
    for X, y in test_a_loader:
        X = X.view(X.size(0), -1)
        scores = model_pool.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

    # Training for concept B
    for X, y in train_b_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_window(X)

        current_model = model_pool.get_model(model_pool.current_model_id)

        if current_model.num_batches > INIT_BATCHES:
            # Concept Drift detection
            if current_model.is_drift():
                logger.info("concept drift detected!")

                detected.append(1)

                adapted_model_id = model_pool.find_adapted_model()
                if adapted_model_id is not None:
                    model_pool.current_model_id = adapted_model_id

                    logger.info(f"find adapted model: {adapted_model_id}")
                else:
                    # Add new model
                    new_model_id = model_pool.add_model()
                    model_pool.current_model_id = new_model_id

                    logger.info(f"add new model: {new_model_id}")

        if not (current_model.num_batches > INIT_BATCHES and current_model.is_drift()):
            detected.append(0)

        # Train current model
        loss = model_pool.stream_train(X)

        losses.append(loss.detach())

    # Evaluation for concept B
    for X, y in test_b_loader:
        X = X.view(X.size(0), -1)
        scores = model_pool.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

    # Training for concept C
    for X, y in train_c_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_window(X)

        current_model = model_pool.get_model(model_pool.current_model_id)

        if current_model.num_batches > INIT_BATCHES:
            # Concept Drift detection
            if current_model.is_drift():
                logger.info("concept drift detected!")

                detected.append(1)

                adapted_model_id = model_pool.find_adapted_model()
                if adapted_model_id is not None:
                    model_pool.current_model_id = adapted_model_id

                    logger.info(f"find adapted model: {adapted_model_id}")
                else:
                    # Add new model
                    new_model_id = model_pool.add_model()
                    model_pool.current_model_id = new_model_id

                    logger.info(f"add new model: {new_model_id}")

        if not (current_model.num_batches > INIT_BATCHES and current_model.is_drift()):
            detected.append(0)

        # Train current model
        loss = model_pool.stream_train(X)

        losses.append(loss.detach())

    # Evaluation for concept C
    for X, y in test_c_loader:
        X = X.view(X.size(0), -1)
        scores = model_pool.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

    print(f"AUROC: {auroc.compute():0.5f}")
    print(f"AUPRC: {auprc.compute():0.5f}")

    losses = np.array(losses)
    detected = np.array(detected)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(losses, color=LOSS_COLOR, label="Losses")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss", color=LOSS_COLOR)
    ax1.tick_params(axis="y", labelcolor=LOSS_COLOR)

    # Create a twin axis for the detected scatter plot
    ax2 = ax1.twinx()

    detected_indices = np.where(detected == 1)[0]
    ax2.scatter(detected_indices, [1] * len(detected_indices), color=DETECTED_COLOR, label="Drift Detected", alpha=0.6)
    ax2.set_ylabel("Drift Detected", color=DETECTED_COLOR)
    ax2.tick_params(axis="y", labelcolor=DETECTED_COLOR)
    ax2.set_yticks([0, 1])
    ax2.set_ylim(-0.1, 1.1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Recurring Concept Drift (Kuzushiji-MNIST)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()