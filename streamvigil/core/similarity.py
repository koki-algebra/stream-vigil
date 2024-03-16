import math

import torch


def _centering(x: torch.Tensor) -> torch.Tensor:
    n = x.size()[0]
    j = torch.ones(n, n)
    i = torch.eye(n)
    h = i - j / n

    return torch.mm(h.mm(x), h)


def _rbf(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """
    Radial Basis Function Kernel (RBF Kernel)
    """
    # Gram matrix
    gx = x.mm(x.T)

    tmp = gx.diag() - gx
    kx = tmp + tmp.T

    if sigma is None:
        mdist = kx[kx != 0].median()
        sigma = math.sqrt(mdist.item())

    kx = (-0.5 / (sigma * sigma)) * kx
    return kx.exp()


def _kernel_HSIC(x1: torch.Tensor, x2: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion (HSIC)
    """
    return (_centering(_rbf(x1, sigma)) * _centering(_rbf(x2, sigma))).sum()


def _linear_HSIC(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion (HSIC)
    """
    lx1 = x1.mm(x1.T)
    lx2 = x2.mm(x2.T)
    return (_centering(lx1) * _centering(lx2)).sum()


def linear_CKA(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Centered Kernel Alignment (CKA)
    """
    hsic = _linear_HSIC(x1, x2)
    var1 = _linear_HSIC(x1, x1).sqrt()
    var2 = _linear_HSIC(x2, x2).sqrt()

    return hsic / (var1 * var2)


def kernel_CKA(x1: torch.Tensor, x2: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """
    Centered Kernel Alignment (CKA)
    """
    hsic = _kernel_HSIC(x1, x2, sigma)
    var1 = _kernel_HSIC(x1, x1, sigma)
    var2 = _kernel_HSIC(x2, x2, sigma)

    return hsic / (var1 * var2)
