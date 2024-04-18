import torch

from streamvigil.core.similarity import _centering, _kernel_HSIC, _linear_HSIC, _rbf, kernel_CKA, linear_CKA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_centering():
    a = 1.0
    b = 2.0
    c = 1.0
    d = 2.0
    x = torch.tensor([[a, b], [c, d]]).to(device)

    got = _centering(x, device)

    exp = torch.tensor([[a - c - b + d, -(a - c - b + d)], [-(a - c - b + d), a - c - b + d]]).to(device) / 4

    torch.testing.assert_close(exp, got)


def test_rbf():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    sigma = 1.0
    got = _rbf(x, sigma)
    assert isinstance(got, torch.Tensor)
    assert got.shape == (2, 2)


def test_kernel_HSIC():
    x1 = torch.randn(100, 64).to(device)
    x2 = torch.randn(100, 64).to(device)
    got = _kernel_HSIC(x1, x2, device)
    assert got.dim() == 0


def test_linear_HSIC():
    x1 = torch.randn(100, 64).to(device)
    x2 = torch.randn(100, 64).to(device)
    got = _linear_HSIC(x1, x2, device)
    assert got.dim() == 0


def test_linear_CKA():
    x1 = torch.randn(100, 64).to(device)
    x2 = torch.randn(100, 64).to(device)
    got = linear_CKA(x1, x2, device)
    assert got.dim() == 0


def test_kernel_CKA():
    x1 = torch.randn(100, 64).to(device)
    x2 = torch.randn(100, 64).to(device)
    got = kernel_CKA(x1, x2, device)
    assert got.dim() == 0
