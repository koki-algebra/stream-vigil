from typing import Dict, List

import torch
import torch.nn as nn

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
