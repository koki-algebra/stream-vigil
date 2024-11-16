from logging import getLogger
from logging.config import dictConfig
from typing import List

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil._arcus_model import ARCUSModel
from streamvigil._arcus_model_pool import ARCUSModelPool
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import filter_index, set_seed, to_anomaly_labels

RANDOM_STATE = 80
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

LOSS_COLOR = "#00ADD8"
RELIABILITY_COLOR = "#00A29C"
DETECTED_COLOR = "#CE3262"


def get_data_loader(
    normal_labels: List[int],
    anomaly_labels: List[int],
    anomaly_ratio=0.01,
    train=True,
):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
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
    model_pool = ARCUSModelPool[ARCUSModel](detector)

    # Data loader
    train_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[7, 8, 9],
        anomaly_ratio=0.001,
    )
    test_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[0, 3, 4, 5, 6, 7, 8, 9],
    )

    auroc = BinaryAUROC()
    auprc = BinaryAUPRC()

    # Train initial model
    init_model_id = model_pool.add_model()

    logger.info("start training the initial model")
    for X, _ in train_loader:
        X = X.view(X.size(0), -1)
        model_pool.stream_train(init_model_id, X)
    logger.info("finish training initial model")

    reliabilities = [model_pool._reliability]
    detected = []

    # Training
    for X, y in train_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_reliability(X)
        reliabilities.append(model_pool._reliability)

        if model_pool.is_drift():
            logger.info("concept drift detected!")
            model_id = model_pool.add_model()

            logger.info(f"new model {model_id} is added")

            model_pool.stream_train(model_id, X)

            if model_pool.compress(X, model_id):
                logger.info("model pool compressed!")

            detected.append(1)
        else:
            model_id = model_pool.find_most_reliable_model()
            model_pool.stream_train(model_id, X)

            detected.append(0)

    # Evaluation
    for X, y in test_loader:
        X = X.view(X.size(0), -1)

        scores = model_pool.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

    print(f"AUROC: {auroc.compute():0.5f}")
    print(f"AUPRC: {auprc.compute():0.5f}")

    plt.figure(figsize=(12, 6))
    plt.plot(reliabilities, label="Reliability", color=RELIABILITY_COLOR)

    detected_indices = [i for i, x in enumerate(detected) if x == 1]

    plt.scatter(detected_indices, [1] * len(detected_indices), label="Detected", color=DETECTED_COLOR)

    plt.xlabel("Iterations")
    plt.ylabel("Model Pool Reliability")
    plt.legend()
    plt.title("No Drift (MNIST)")
    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
