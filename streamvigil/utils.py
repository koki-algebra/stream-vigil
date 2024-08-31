import random
from copy import deepcopy
from typing import List

import numpy as np
import torch
from torchvision.datasets import MNIST


def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_by_label(
    dataset: MNIST,
    normal_labels: List[int],
    anomaly_labels: List[int],
    anomaly_ratio: float = 0.05,
):
    normal_indices = [i for i, target in enumerate(dataset.targets) if target in normal_labels]
    anomaly_indices = [i for i, target in enumerate(dataset.targets) if target in anomaly_labels]

    total_samples = len(normal_indices) + len(anomaly_indices)
    desired_anomaly_samples = int(total_samples * anomaly_ratio)

    if len(anomaly_indices) > desired_anomaly_samples:
        anomaly_indices = random.sample(anomaly_indices, desired_anomaly_samples)

    selected_indices = normal_indices + anomaly_indices

    filtered_dataset = deepcopy(dataset)
    filtered_dataset.data = filtered_dataset.data[selected_indices]
    filtered_dataset.targets = filtered_dataset.targets[selected_indices]

    # ラベルを二値分類用に変更（0: normal, 1: anomaly）
    filtered_dataset.targets = torch.tensor(
        [0 if target in normal_labels else 1 for target in filtered_dataset.targets]
    )

    return filtered_dataset
