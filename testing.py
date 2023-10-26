"""Functions for unit tests."""

import torch
from vector_functions import ensure_tensor, norm


torch.set_default_tensor_type('torch.DoubleTensor')


machine_epsilon_f64 = 2**-52


def comparetensors(a, b, error=1000.0, error_in_meps=True):
    """
    Check whether torch tensors are of equal value within defined error range.

    Required precision of computed answers: error x float64 (a.k.a. double) machine epsilon.
    """
    if error_in_meps:
        margin = error * machine_epsilon_f64
    else:
        margin = error

    return torch.all(torch.abs(a - b) < margin)


def checkunitvector(a, error=1000):
    """Check wether tensor is a unit vector (or NaN)."""
    # return comparetensors(norm(a).nan_to_num(nan=1), 1, error)
    return comparetensors(norm(a), 1, error)


def MSE(a, b, dim=()):
    """Mean Square Error."""
    return (a - b).abs().square().mean(dim=dim)


def SSE(a, b, dim=()):
    """Sum Square Error."""
    return (a - b).abs().square().sum(dim=dim)


def weighted_MSE(a, b, w, dim=()):
    """Weighted Mean Square Error."""
    return weighted_mean((a - b).abs().square(), w, dim=dim)


def weighted_SSE(a, b, w, dim=()):
    """Weighted Sum Square Error."""
    return ((w * (a - b)).abs().square()).sum(dim=dim)


def weighted_mean(a_in, w_in, dim=()):
    """Weighted Mean."""
    a = ensure_tensor(a_in)
    w = ensure_tensor(w_in)
    assert (a.shape == w.shape) or (a.numel() == 1 and w.numel() == 1)
    return torch.sum(a * w, dim=dim) / torch.sum(w, dim=dim)
