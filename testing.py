"""Functions for unit tests."""

import torch
from vector_functions import norm


def comparetensors(a, b, error=100):
    """
    Check whether torch tensors are of equal value within defined error range.

    Required precision of computed answers: error x float32-epsilon.
    """
    return torch.all(torch.abs(a - b) < (error * (2**-23)))


def checkunitvector(a, error=100):
    """Check wether tensor is a unit vector."""
    return comparetensors(norm(a), 1, error)


def MSE(a, b):
    """Mean Square Error (without NaNs)."""
    return (a - b).nan_to_num(nan=0.0).abs().square().mean()