from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ADBenchDataset(Dataset):
    def __init__(
        self, filepath: str, train=True, test_size=0.3, shuffle=False, random_state: float | None = None
    ) -> None:
        super().__init__()

        data = np.load(filepath, allow_pickle=True)

        X_train, X_test, y_train, y_test = train_test_split(
            data["X"], data["y"], test_size=test_size, shuffle=shuffle, random_state=random_state
        )

        if train:
            self.X = torch.from_numpy(X_train.astype(np.float32))
            self.y = torch.from_numpy(y_train.astype(np.float32))
        else:
            self.X = torch.from_numpy(X_test.astype(np.float32))
            self.y = torch.from_numpy(y_test.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
