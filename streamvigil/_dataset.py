import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, csv_file: str, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.data.iloc[idx, :-1]
        labels = self.data.iloc[idx, -1]

        if self.transform:
            features = self.transform(features)

        if self.target_transform:
            labels = self.target_transform(labels)

        features = torch.tensor(features.values, dtype=torch.float)

        return features, labels
