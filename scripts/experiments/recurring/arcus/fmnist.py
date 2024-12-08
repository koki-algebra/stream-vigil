from logging import getLogger
from logging.config import dictConfig
from typing import List

from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil._arcus_model import ARCUSModel
from streamvigil._arcus_model_pool import ARCUSModelPool
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
    dataset = datasets.FashionMNIST(
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
        batch_size,
        shuffle=True,
    )

    return loader


def run(
    random_state=80,
    train_batch_size=128,
    test_batch_size=128,
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
    model_pool = ARCUSModelPool[ARCUSModel](detector)

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

    reliabilities = [model_pool._reliability]
    detected = []
    num_models = []

    # Training for concept A
    for X, y in train_a_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_reliability(X)
        reliabilities.append(model_pool._reliability)

        num_models.append(len(model_pool.get_models()))

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

    # Evaluation for concept A
    for X, y in test_a_loader:
        X = X.view(X.size(0), -1)

        scores = model_pool.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

    # Training for concept B
    for X, y in train_b_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_reliability(X)
        reliabilities.append(model_pool._reliability)

        num_models.append(len(model_pool.get_models()))

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

    # Evaluation for concept B
    for X, y in test_b_loader:
        X = X.view(X.size(0), -1)

        scores = model_pool.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

    # Training for concept C
    for X, y in train_c_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_reliability(X)
        reliabilities.append(model_pool._reliability)

        num_models.append(len(model_pool.get_models()))

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

    # Evaluation for concept C
    for X, y in test_c_loader:
        X = X.view(X.size(0), -1)

        scores = model_pool.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

    result = {
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "reliabilities": reliabilities,
        "detected": detected,
        "num_models": num_models,
    }

    return result
