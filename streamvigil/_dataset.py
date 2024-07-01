from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ADBenchDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        train=True,
        test_size=0.3,
        shuffle=False,
        random_state: float | None = None,
    ) -> None:
        super().__init__()

        data = np.load(filepath, allow_pickle=True)

        X_train, X_test, y_train, y_test = train_test_split(
            data["X"],
            data["y"],
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
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


class CSVDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        train=True,
        test_size=0.3,
        labeled_size=1.0,
        shuffle=False,
        random_state: float | None = None,
    ) -> None:
        super().__init__()

        data = pd.read_csv(filepath)
        X_train = data.iloc[:, :-1].to_numpy()
        y_train = data.iloc[:, -1].to_numpy()

        if test_size != 0.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train,
                y_train,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state,
            )

        if train:
            self.X = torch.from_numpy(X_train.astype(np.float32))
            self.y = torch.from_numpy(y_train.astype(np.float32))
            if labeled_size >= 0.0 and labeled_size < 1.0:
                unlabeled_indices = np.random.choice(
                    len(X_train),
                    size=int((1.0 - labeled_size) * len(X_train)),
                    replace=False,
                )
                self.y[unlabeled_indices] = torch.nan
        else:
            self.X = torch.from_numpy(X_test.astype(np.float32))
            self.y = torch.from_numpy(y_test.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
