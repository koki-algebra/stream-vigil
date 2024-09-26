import random
from copy import deepcopy
from typing import List

import torch
from torchvision.datasets import MNIST


def filter_mnist(
    dataset: MNIST,
    normal_labels: List[int],
    anomaly_labels: List[int],
    anomaly_ratio: float = 0.01,
):
    selected_idx = filter_index(
        dataset.targets,
        normal_labels,
        anomaly_labels,
        anomaly_ratio,
    )

    filtered_dataset = deepcopy(dataset)
    filtered_dataset.data = filtered_dataset.data[selected_idx]
    filtered_dataset.targets = filtered_dataset.targets[selected_idx]

    # ラベルを二値分類用に変更（0: normal, 1: anomaly）
    filtered_dataset.targets = to_anomaly_labels(filtered_dataset.targets, normal_labels)

    return filtered_dataset


def filter_index(
    origin_labels: torch.Tensor,
    normal_labels: List[int],
    anomaly_labels: List[int],
    anomaly_ratio=0.01,
):
    normal_idx = torch.where(torch.isin(origin_labels, torch.tensor(normal_labels)))[0].tolist()
    anomaly_idx = torch.where(torch.isin(origin_labels, torch.tensor(anomaly_labels)))[0].tolist()

    total_samples = len(normal_idx) + len(anomaly_idx)
    desired_anomaly_samples = int(total_samples * anomaly_ratio)

    if len(anomaly_idx) > desired_anomaly_samples:
        anomaly_idx = random.sample(anomaly_idx, desired_anomaly_samples)

    selected_idx = normal_idx + anomaly_idx
    random.shuffle(selected_idx)

    return selected_idx


def to_anomaly_labels(
    origin_labels: torch.Tensor,
    normal_labels: List[int],
):
    return torch.tensor([0 if origin_label in normal_labels else 1 for origin_label in origin_labels])
