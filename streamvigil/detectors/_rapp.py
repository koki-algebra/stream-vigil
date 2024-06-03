from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from streamvigil.core import AnomalyDetector

from ._basic import BasicAutoEncoder

ENCODER_LAST_LAYER = "layer_last"


class _RAPP(BasicAutoEncoder):
    def __init__(self, encoder_dims: List[int], decoder_dims: List[int], batch_norm=False) -> None:
        super().__init__(encoder_dims, decoder_dims, batch_norm)

        # Register forward hooks
        self._encoder_activations = self._register_hooks(self.encoder)
        self._decoder_activations = self._register_hooks(self.decoder)

    def _register_hooks(self, network: nn.Module) -> Dict[str, torch.Tensor]:
        activations: Dict[str, torch.Tensor] = {}
        layer_cnt = 0
        for idx, layer in enumerate(network):
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(self._get_activation_hook(activations, f"layer_{layer_cnt}"))
                layer_cnt += 1
            elif idx == len(network) - 1:
                layer.register_forward_hook(self._get_activation_hook(activations, ENCODER_LAST_LAYER))

        return activations

    def _get_activation_hook(self, activations: Dict[str, torch.Tensor], name: str):
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            activations[name] = output.detach()

        return hook

    def get_encoder_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get the dictionary containing the output of each layer of the encoder.

        Returns
        -------
        activations : Dict[str, torch.Tensor]
            dict[layer_name, hidden representations]
        """
        return self._encoder_activations

    def get_decoder_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get the dictionary containing the output of each layer of the decoder.

        Returns
        -------
        activations : Dict[str, torch.Tensor]
            dict[layer_name, hidden representations]
        """
        return self._decoder_activations

    def get_latent(self) -> torch.Tensor | None:
        return self._encoder_activations.get(ENCODER_LAST_LAYER)


class RAPP(AnomalyDetector):
    def __init__(
        self,
        encoder_dims: List[int],
        decoder_dims: List[int],
        batch_norm=False,
        learning_rate=0.0001,
    ) -> None:
        auto_encoder = _RAPP(
            encoder_dims,
            decoder_dims,
            batch_norm=batch_norm,
        )
        super().__init__(auto_encoder, learning_rate)
        self._auto_encoder = auto_encoder

        self._criterion = nn.MSELoss()

    def _sap(self, h_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Simple Aggregation along Pathway (SAP)

        Returns
        -------
        score : float
            Novelty score
        """
        if len(h_pairs) == 0:
            raise ValueError("h_pairs must have length greater than or equal to 1")
        score = torch.zero_(h_pairs[0][0])
        for h, h_pred in h_pairs:
            score += (h - h_pred).norm(dim=1).square()
        return score

    def _nap(self, h_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Normalized Aggregation along Pathway (NAP)
        """

        for h, h_pred in h_pairs:
            d = h - h_pred
            d = d - d.mean(dim=0)

        _, s, v = torch.linalg.svd(d, full_matrices=False)
        s[s == 0] = 1.0

        return d.matmul(v.T).matmul(torch.linalg.inv(s.diag())).norm(dim=1).square()

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
        x_pred: torch.Tensor = self._auto_encoder(x)

        x_activations = self._auto_encoder.get_encoder_activations().copy()

        self._auto_encoder(x_pred)

        x_pred_activations = self._auto_encoder.get_encoder_activations().copy()

        h_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for h, h_pred in zip(list(x_activations.values()), list(x_pred_activations.values())):
            h_pairs.append((h, h_pred))

        return self._nap(h_pairs)