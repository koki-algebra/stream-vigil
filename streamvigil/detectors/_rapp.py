from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from streamvigil.core import AnomalyDetector

from ._basic import BasicAutoEncoder

ENCODER_LAST_LAYER = "layer_last"


class _SaveActivations:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._activations: Dict[str, torch.Tensor] = {}
        self._hook_list: List[RemovableHandle] = []

        self._registor_model(model)

    def __enter__(self):
        return self

    def __call__(self, model):
        self.__init__(model)

    def __exit__(self, exc_type, exc_value, traceback):
        self._remove_hook()
        self._clear_buffer()

    def _clear_buffer(self):
        self._activations = {}

    def _registor_model(self, model):
        layer_cnt = 0
        for idx, layer in enumerate(model):
            if isinstance(layer, nn.ReLU):
                handle = layer.register_forward_hook(self._save(f"layer_{layer_cnt}"))
                self._hook_list.append(handle)
                layer_cnt += 1
            elif idx == len(model) - 1:
                handle = layer.register_forward_hook(self._save(ENCODER_LAST_LAYER))
                self._hook_list.append(handle)

    def _remove_hook(self):
        for handle in self._hook_list:
            handle.remove()

    def _save(self, name: str):
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self._activations[name] = output.detach()

        return hook

    def get_activations(self):
        return self._activations

    def get_latent(self) -> torch.Tensor:
        return self._activations[ENCODER_LAST_LAYER]


class RAPP(AnomalyDetector):
    def __init__(
        self,
        encoder_dims: List[int],
        decoder_dims: List[int],
        batch_norm=False,
        learning_rate=0.0001,
    ) -> None:
        auto_encoder = BasicAutoEncoder(
            encoder_dims,
            decoder_dims,
            batch_norm=batch_norm,
        )
        super().__init__(auto_encoder, learning_rate)

        self._criterion = nn.MSELoss()

    def _sap(self, h_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Simple Aggregation along Pathway (SAP)

        Returns
        -------
        score : float
            Novelty score
        """
        score = torch.tensor(0.0)
        for h, h_pred in h_pairs:
            score = (h - h_pred).norm(dim=1).square() + score
        return score

    def _nap(self, h_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Normalized Aggregation along Pathway (NAP)
        """

        diffs = []
        for h, h_pred in h_pairs:
            diffs.append(h - h_pred)

        D = torch.concat(diffs, dim=1)

        D = D - D.mean(dim=0)
        _, S, V = torch.linalg.svd(D, full_matrices=False)

        S[S == 0] = 1.0
        S_inv = torch.linalg.inv(S.diag())

        return D.matmul(V.T).matmul(S_inv).norm(dim=1).square()

    def train(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        optimizer = self._load_optimizer()
        self._auto_encoder.train()

        x_pred: torch.Tensor = self._auto_encoder(x)

        loss: torch.Tensor = self._criterion(x, x_pred)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        h_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        with _SaveActivations(self._auto_encoder.encoder) as sa:
            x_pred: torch.Tensor = self._auto_encoder(x)
            x_activations = sa.get_activations()

        with _SaveActivations(self._auto_encoder.encoder) as sa:
            self._auto_encoder(x_pred)
            x_pred_activations = sa.get_activations()

        for h, h_pred in zip(list(x_activations.values()), list(x_pred_activations.values())):
            h_pairs.append((h, h_pred))

        return self._nap(h_pairs)
