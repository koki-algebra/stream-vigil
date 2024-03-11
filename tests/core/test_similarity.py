import torch

from streamvigil.core._similarity import _centering, _kernel_HSIC, _linear_HSIC, _rbf, kernel_CKA, linear_CKA


def test_centering():
    a = 1.0
    b = 2.0
    c = 1.0
    d = 2.0
    x = torch.tensor([[a, b], [c, d]])

    got = _centering(x)

    exp = torch.tensor([[a - c - b + d, -(a - c - b + d)], [-(a - c - b + d), a - c - b + d]]) / 4

    torch.testing.assert_close(exp, got)


def test_rbf():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    sigma = 1.0
    got = _rbf(x, sigma)
    assert isinstance(got, torch.Tensor)
    assert got.shape == (2, 2)


def test_kernel_HSIC():
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    got = _kernel_HSIC(x1, x2)
    assert got.dim() == 0


def test_linear_HSIC():
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    got = _linear_HSIC(x1, x2)
    assert got.dim() == 0


def test_linear_CKA():
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    got = linear_CKA(x1, x2)
    assert got.dim() == 0


def test_kernel_CKA():
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    got = kernel_CKA(x1, x2)
    assert got.dim() == 0
