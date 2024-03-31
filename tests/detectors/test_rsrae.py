import torch

from streamvigil.detectors import RSRAE


def test_rsrae():
    rsrae = RSRAE(encoder_dims=[8, 16, 32, 64, 128], rsr_dim=4, decoder_dims=[4, 6, 8])
    x = torch.randn(32, 8)
    rsrae(x)
