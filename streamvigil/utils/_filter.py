import random
from copy import deepcopy
from typing import List

import torch
from torchvision.datasets import MNIST


def filter_by_label(
    dataset: MNIST,
    normal_labels: List[int],
    anomaly_labels: List[int],
    anomaly_ratio: float = 0.05,
):
    normal_idx = torch.where(torch.isin(dataset.targets, torch.tensor(normal_labels)))[0].tolist()
    anomaly_idx = torch.where(torch.isin(dataset.targets, torch.tensor(anomaly_labels)))[0].tolist()

    total_samples = len(normal_idx) + len(anomaly_idx)
    desired_anomaly_samples = int(total_samples * anomaly_ratio)

    if len(anomaly_idx) > desired_anomaly_samples:
        anomaly_idx = random.sample(anomaly_idx, desired_anomaly_samples)

    selected_indices = normal_idx + anomaly_idx

    filtered_dataset = deepcopy(dataset)
    filtered_dataset.data = filtered_dataset.data[selected_indices]
    filtered_dataset.targets = filtered_dataset.targets[selected_indices]

    # ラベルを二値分類用に変更（0: normal, 1: anomaly）
    filtered_dataset.targets = torch.tensor(
        [0 if target in normal_labels else 1 for target in filtered_dataset.targets]
    )

    return filtered_dataset
