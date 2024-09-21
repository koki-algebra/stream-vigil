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
    normal_idx = torch.where(torch.isin(dataset.targets, torch.tensor(normal_labels)))[0]
    anomaly_idx = torch.where(torch.isin(dataset.targets, torch.tensor(anomaly_labels)))[0]

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
