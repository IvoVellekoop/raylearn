"""Functions for unit tests."""

import torch


def comparetensors(a, b, error=100):
    """
    Check wether torch tensors are of equal value within defined error range.
    Required precision of computed answers: errorx float32 epsilon.
    """
    return torch.all(torch.abs(a - b) < (error * (2**-23)))
