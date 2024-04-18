import math

import torch


def _centering_matrix(n: int) -> torch.Tensor:
    return torch.eye(n) - (1 / n) * torch.ones(n, n)


def _centering(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    n = x.size()[0]
    h = _centering_matrix(n).to(device)

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


def _kernel_HSIC(x1: torch.Tensor, x2: torch.Tensor, device: torch.device, sigma: float | None = None) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion (HSIC)
    """
    if x1.shape != x2.shape:
        raise ValueError("The shapes of x1 and x2 must be match")

    x1 = x1.to(device)
    x2 = x2.to(device)

    # Centered gram matrix
    g1 = _centering(_rbf(x1, sigma), device)
    g2 = _centering(_rbf(x2, sigma), device)

    return g1.mm(g2).norm()


def _linear_HSIC(x1: torch.Tensor, x2: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion (HSIC)
    """
    if x1.shape != x2.shape:
        raise ValueError("The shapes of x1 and x2 must be match")

    x1 = x1.to(device)
    x2 = x2.to(device)

    def kernel(x: torch.Tensor) -> torch.Tensor:
        return x.mm(x.T)

    # Centered gram matrix
    g1 = _centering(kernel(x1), device)
    g2 = _centering(kernel(x2), device)

    return g1.mm(g2).norm()


def linear_CKA(x1: torch.Tensor, x2: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Centered Kernel Alignment (CKA)
    """
    # Small value to avoid division by zero
    eps = 1e-8

    x1 = x1.to(device)
    x2 = x2.to(device)

    covar = _linear_HSIC(x1, x2, device)
    var1 = _linear_HSIC(x1, x1, device).sqrt()
    var2 = _linear_HSIC(x2, x2, device).sqrt()
    return covar / max((var1 * var2).item(), eps)


def kernel_CKA(x1: torch.Tensor, x2: torch.Tensor, device: torch.device, sigma: float | None = None) -> torch.Tensor:
    """
    Centered Kernel Alignment (CKA)
    """
    # Small value to avoid division by zero. Default: 1e-8
    eps = 1e-8

    x1 = x1.to(device)
    x2 = x2.to(device)

    covar = _kernel_HSIC(x1, x2, device, sigma)
    var1 = _kernel_HSIC(x1, x1, device, sigma).sqrt()
    var2 = _kernel_HSIC(x2, x2, device, sigma).sqrt()

    return covar / max((var1 * var2).item(), eps)
