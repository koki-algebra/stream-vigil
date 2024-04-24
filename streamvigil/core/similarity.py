import math

import torch


def _centering_matrix(n: int) -> torch.Tensor:
    return torch.eye(n) - (1 / n) * torch.ones(n, n)


def _centering(x: torch.Tensor) -> torch.Tensor:
    n = x.size()[0]
    h = _centering_matrix(n).to(x.device)

    return torch.mm(h.mm(x), h)


def _rbf(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """
    Radial Basis Function Kernel (RBF Kernel)
    """
    # Small value to avoid division by zero
    eps = 1e-8

    # Gram matrix
    gx = x.mm(x.T)

    tmp = gx.diag() - gx
    kx = tmp + tmp.T

    if sigma is None:
        mdist = kx[kx != 0].median()
        sigma = math.sqrt(mdist.item())

    kx = (-0.5 / max(sigma * sigma, eps)) * kx
    return kx.exp()


def _kernel_HSIC(x1: torch.Tensor, x2: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion (HSIC)
    """
    if x1.shape != x2.shape:
        raise ValueError("The shapes of x1 and x2 must be match")

    # Centered gram matrix
    g1 = _centering(_rbf(x1, sigma)).to(x1.device)
    g2 = _centering(_rbf(x2, sigma)).to(x2.device)

    return g1.mm(g2).norm()


def _linear_HSIC(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion (HSIC)
    """
    if x1.shape != x2.shape:
        raise ValueError("The shapes of x1 and x2 must be match")

    def kernel(x: torch.Tensor) -> torch.Tensor:
        return x.mm(x.T)

    # Centered gram matrix
    g1 = _centering(kernel(x1)).to(x1.device)
    g2 = _centering(kernel(x2)).to(x2.device)

    return g1.mm(g2).norm()


def linear_CKA(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Centered Kernel Alignment (CKA)
    """
    # Small value to avoid division by zero
    eps = 1e-8

    covar = _linear_HSIC(x1, x2)
    var1 = _linear_HSIC(x1, x1).sqrt()
    var2 = _linear_HSIC(x2, x2).sqrt()
    return covar / max((var1 * var2).item(), eps)


def kernel_CKA(x1: torch.Tensor, x2: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """
    Centered Kernel Alignment (CKA)
    """
    # Small value to avoid division by zero. Default: 1e-8
    eps = 1e-8

    covar = _kernel_HSIC(x1, x2, sigma)
    var1 = _kernel_HSIC(x1, x1, sigma).sqrt()
    var2 = _kernel_HSIC(x2, x2, sigma).sqrt()

    return covar / max((var1 * var2).item(), eps)
