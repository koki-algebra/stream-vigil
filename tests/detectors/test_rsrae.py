import torch

from streamvigil.detectors._rsrae import _RSRAE, RSR, RSRAE


def test_rsr():
    rsr = RSR(8, 4)
    x = torch.randn(32, 8)
    rsr(x)


def test_rsrae_network():
    rsrae = _RSRAE(encoder_dims=[8, 16, 32, 64, 128], rsr_dim=4, decoder_dims=[4, 6, 8])
    x = torch.randn(32, 8)
    rsrae(x)


def test_rsrae_loss():
    rsrae = RSRAE(encoder_dims=[8, 16, 32, 64, 128], rsr_dim=4, decoder_dims=[4, 6, 8])

    x = torch.randn(32, 8)
    x_pred = torch.randn(32, 8)
    z = torch.randn(32, 128)

    # loss
    rsrae._reconstruct_loss(x, x_pred)
    rsrae._pca_loss(z)
    rsrae._project_loss()


def test_rsrae_train():
    rsrae = RSRAE(encoder_dims=[8, 16, 32, 64, 128], rsr_dim=4, decoder_dims=[4, 6, 8])
    x = torch.randn(256, 8)
    rsrae.train(x)


def test_rsrae_predict():
    rsrae = RSRAE(encoder_dims=[8, 16, 32, 64, 128], rsr_dim=4, decoder_dims=[4, 6, 8])
    x = torch.randn(5, 8)
    rsrae.predict(x)
