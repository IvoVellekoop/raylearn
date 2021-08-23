"""Test vector functions."""

import torch
from torch import Tensor, stack

from testing import comparetensors
from interpolate import interpolate2d


def test_interpolate1():
    """
    Create an example with b_A, b_D, c_A and c_D set.
    These can be checked inside the interpolate2d function.
    """

    # Relative output coord distances wrt input coords
    b_A = 1/3
    c_A = 1/4
    b_D = 1/5
    c_D = 1/8

    # Input coordinates
    A = Tensor((1, 0))
    B = Tensor((3, 1))
    C = Tensor((0, 2))
    D = Tensor((2, 3))

    # Values at input coordinates
    VA = 1
    VB = 2
    VC = 9
    VD = 6

    # Compute coordinates and values
    coords_in = stack((stack((A, B)), stack((C, D))))
    coords_out = stack((A + b_A*(B-A) + c_A*(C-A),
                        D + b_D*(B-D) + c_D*(C-D))).view(1, 2, 2)

    values_in = Tensor(((VA, VB), (VC, VD))).view(2, 2, 1)
    values_out_man = Tensor((VA + b_A*(VB-VA) + c_A*(VC-VA),
                             VD + b_D*(VB-VD) + c_D*(VC-VD))).view(1, 2, 1)

    # Interpolate
    values_out_unfolded, mask = interpolate2d(coords_in, values_in, coords_out)
    values_out = torch.sum(mask * values_out_unfolded, dim=(0, 1, 2))

    assert comparetensors(values_out_man, values_out)
    assert torch.all(torch.tensor(values_out_unfolded.shape) == Tensor((2, 1, 1, 1, 2, 1)))
