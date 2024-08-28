import random
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from streamvigil import PerformanceBaseModelPool
from streamvigil.core import Model
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import set_seed

train_batch_size = 128
test_batch_size = 64
epochs = 5


def filter_by_label(normal_labels: List[int], anomaly_labels: List[int], anomaly_ratio: float = 0.05, is_train=True):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(
        root="./data/pytorch",
        train=is_train,
        download=True,
        transform=transform,
    )

    normal_indices = [i for i, target in enumerate(dataset.targets) if target in normal_labels]
    anomaly_indices = [i for i, target in enumerate(dataset.targets) if target in anomaly_labels]

    total_samples = len(normal_indices) + len(anomaly_indices)
    desired_anomaly_samples = int(total_samples * anomaly_ratio)

    if len(anomaly_indices) > desired_anomaly_samples:
        anomaly_indices = random.sample(anomaly_indices, desired_anomaly_samples)

    selected_indices = normal_indices + anomaly_indices

    dataset.data = dataset.data[selected_indices]
    dataset.targets = dataset.targets[selected_indices]

    # ラベルを二値分類用に変更（0: normal, 1: anomaly）
    dataset.targets = torch.tensor([0 if target in normal_labels else 1 for target in dataset.targets])

    return dataset


def train(dataset: datasets.MNIST, model: Model):
    loader = DataLoader(
        dataset,
        batch_size=train_batch_size,
    )

    size = len(loader.dataset)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        for batch, (X, _) in enumerate(loader):
            X = X.view(X.size(0), -1)
            loss = model.stream_train(X)

            if batch % 100 == 0:
                loss = loss.item()
                current = batch * train_batch_size + len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def main():
    random_state = 90
    set_seed(random_state)

    # Dataset
    concept_a_dataset = filter_by_label(normal_labels=[1, 2], anomaly_labels=[3])
    concept_b_dataset = filter_by_label(normal_labels=[4, 5], anomaly_labels=[6])
    concept_c_dataset = filter_by_label(normal_labels=[7, 8], anomaly_labels=[9])

    auto_encoder = BasicAutoEncoder(
        encoder_dims=[784, 588, 392, 196],
        decoder_dims=[196, 392, 588, 784],
        batch_norm=True,
    )
    detector = BasicDetector(auto_encoder)

    model_pool = PerformanceBaseModelPool(detector)

    # Add models
    model_id_a = model_pool.add_model()
    model_id_b = model_pool.add_model()
    model_id_c = model_pool.add_model()

    model_a = model_pool.get_model(model_id_a)
    model_b = model_pool.get_model(model_id_b)
    model_c = model_pool.get_model(model_id_c)

    # Training
    print("start training model A")
    train(concept_a_dataset, model_a)
    print("start training model B")
    train(concept_b_dataset, model_b)
    print("start training model C")
    train(concept_c_dataset, model_c)

    # Test
    concept_a_test_dataset = filter_by_label(normal_labels=[1, 2], anomaly_labels=[3], is_train=False)
    concept_b_test_dataset = filter_by_label(normal_labels=[4, 5], anomaly_labels=[6], is_train=False)
    concept_c_test_dataset = filter_by_label(normal_labels=[7, 8], anomaly_labels=[9], is_train=False)

    concept_a_test_loader = DataLoader(
        concept_a_test_dataset,
        batch_size=test_batch_size,
    )
    concept_b_test_loader = DataLoader(
        concept_b_test_dataset,
        batch_size=test_batch_size,
    )
    concept_c_test_loader = DataLoader(
        concept_c_test_dataset,
        batch_size=test_batch_size,
    )

    test_total = 0
    correct = 0
    for X, y in concept_a_test_loader:
        X = X.view(X.size(0), -1)
        test_total += 1
        model_id = model_pool.select_model(X, y)
        if model_id == model_id_a:
            correct += 1

    for X, y in concept_b_test_loader:
        X = X.view(X.size(0), -1)
        test_total += 1
        model_id = model_pool.select_model(X, y)
        if model_id == model_id_b:
            correct += 1

    for X, y in concept_c_test_loader:
        X = X.view(X.size(0), -1)
        test_total += 1
        model_id = model_pool.select_model(X, y)
        if model_id == model_id_c:
            correct += 1

    print(f"correct: {correct}, total: {test_total}")
    print(f"accuracy: {correct / test_total}")


if __name__ == "__main__":
    main()
