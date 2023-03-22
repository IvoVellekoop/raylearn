"""Functions for unit tests."""

import torch
from vector_functions import norm


torch.set_default_tensor_type('torch.DoubleTensor')


machine_epsilon_f64 = 2**-52


def comparetensors(a, b, error=1000):
    """
    Check whether torch tensors are of equal value within defined error range.

    Required precision of computed answers: error x float64 (a.k.a. double) machine epsilon.
    """
    # return torch.all(torch.abs(a - b) < (error * machine_epsilon_f64))
    return torch.all(torch.abs(a - b).nan_to_num(nan=0.0) < (error * machine_epsilon_f64))


def checkunitvector(a, error=1000):
    """Check wether tensor is a unit vector (or NaN)."""
    # return comparetensors(norm(a).nan_to_num(nan=1), 1, error)
    return comparetensors(norm(a), 1, error)


def MSE(a, b, dim=()):
    """Mean Square Error (without NaNs)."""
    return (a - b).nan_to_num(nan=0.0).abs().square().mean(dim=dim)


def weighted_MSE(a, b, w, dim=()):
    """Weighted Mean Square Error (without NaNs)."""
    return weighted_mean((a - b).abs().square(), w, dim=dim)


def weighted_mean(a, w, dim=()):
    """Weighted Mean (without NaNs)."""
    assert a.shape == w.shape
    # return torch.sum(a.nan_to_num(nan=0.0) * w, dim=dim) / torch.sum(w, dim=dim)
    return torch.sum(a * w, dim=dim) / torch.sum(w, dim=dim)
