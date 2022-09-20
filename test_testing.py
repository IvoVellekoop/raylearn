"""Test testing."""

import torch
from torch import Tensor

from testing import comparetensors, checkunitvector, MSE, weighted_mean


torch.set_default_tensor_type('torch.DoubleTensor')


def test_weighted_mean1():
    """Test weighted mean with multiple dimensions."""
    a = Tensor((4, 7))                      # Values
    w = Tensor((2, 1))                      # Weights
    WM = weighted_mean(a, w)                # Weighted Mean
    assert comparetensors(5, WM)


def test_weighted_mean2():
    """Test weighted mean with multiple dimensions."""
    a = Tensor(((1, 2, 3), (1, 2, 3)))      # Values
    w = Tensor(((2, 2, 1), (2, 2, 4)))      # Weights

    WM1 = weighted_mean(a, w, dim=0)        # Weighted Mean over dimension
    assert comparetensors(WM1, Tensor((1, 2, 3)))

    WM2 = weighted_mean(a, w, dim=1)        # Weighted Mean over dimension
    assert comparetensors(WM2, Tensor((1.8, 2.25)))
