from logging import getLogger
from logging.config import dictConfig
from typing import List

from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil.core import Model, ModelPool
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import filter_index, set_seed, to_anomaly_labels


def get_data_loader(
    normal_labels: List[int],
    anomaly_labels: List[int],
    batch_size=128,
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
        batch_size=batch_size,
        shuffle=True,
    )

    return loader


def run(
    random_state=80,
    train_batch_size=128,
    test_batch_size=128,
    latest_window_size=500,
    historical_window_size=500,
    last_window_size=500,
    window_gap=500,
    alpha=0.05,
    init_batches=50,
):
    set_seed(random_state)

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
        historical_window_size=historical_window_size,
        latest_window_size=latest_window_size,
        last_trained_size=last_window_size,
        window_gap=window_gap,
        alpha=alpha,
    )

    # Data loader
    train_a_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[7, 8, 9],
        batch_size=train_batch_size,
        anomaly_ratio=0.001,
    )
    train_b_loader = get_data_loader(
        normal_labels=[3, 4],
        anomaly_labels=[7, 8, 9],
        batch_size=train_batch_size,
        anomaly_ratio=0.001,
    )
    train_c_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[7, 8, 9],
        batch_size=train_batch_size,
        anomaly_ratio=0.001,
    )
    test_a_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[0, 3, 4, 5, 6, 7, 8, 9],
        batch_size=test_batch_size,
        train=False,
    )
    test_b_loader = get_data_loader(
        normal_labels=[3, 4],
        anomaly_labels=[0, 5, 6, 7, 8, 9],
        batch_size=test_batch_size,
        train=False,
    )
    test_c_loader = get_data_loader(
        normal_labels=[1, 2],
        anomaly_labels=[0, 5, 6, 7, 8, 9],
        batch_size=test_batch_size,
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

        if current_model.num_batches > init_batches:
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

        if not (current_model.num_batches > init_batches and current_model.is_drift()):
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

        if current_model.num_batches > init_batches:
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

        if not (current_model.num_batches > init_batches and current_model.is_drift()):
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

        if current_model.num_batches > init_batches:
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

        if not (current_model.num_batches > init_batches and current_model.is_drift()):
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

    result = {
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "losses": losses,
        "detected": detected,
    }

    return result
