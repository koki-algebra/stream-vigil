from logging import getLogger
from logging.config import dictConfig

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
TRAIN_BATCH_SIZE = 128
INIT_BATCHES = 20

LOSS_COLOR = "#00ADD8"
DETECTED_COLOR = "#CE3262"


def main():
    set_seed(RANDOM_STATE)

    with open("./notebooks/logging.yml", encoding="utf-8") as file:
        config = safe_load(file)
    dictConfig(config)
    logger = getLogger(__name__)

    # Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(
        root="./data/pytorch",
        train=True,
        download=True,
        transform=transform,
    )
    concept_a_idx = filter_index(
        train_dataset.targets,
        normal_labels=[1, 2],
        anomaly_labels=[7, 8, 9],
    )
    concept_b_idx = filter_index(
        train_dataset.targets,
        normal_labels=[3, 4],
        anomaly_labels=[7, 8, 9],
    )
    train_dataset.targets[concept_a_idx] = to_anomaly_labels(
        train_dataset.targets[concept_a_idx],
        normal_labels=[1, 2],
    )
    train_dataset.targets[concept_b_idx] = to_anomaly_labels(
        train_dataset.targets[concept_b_idx],
        normal_labels=[3, 4],
    )

    result_idx = concept_a_idx + concept_b_idx
    train_dataset.data = train_dataset.data[result_idx]
    train_dataset.targets = train_dataset.targets[result_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
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

    auroc = BinaryAUROC()
    auprc = BinaryAUPRC()

    losses = []
    detected = []

    # Training
    for X, y in train_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_window(X)

        current_model = model_pool.get_model(model_pool.current_model_id)

        scores = current_model.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

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

    losses = np.array(losses)
    detected = np.array(detected)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    print(f"AUROC: {auroc.compute():0.5f}")
    print(f"AUPRC: {auprc.compute():0.5f}")

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

    plt.title("Abrupt Drift (Fashion-MNIST)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
