"""
Test math functions
"""

from torch import Tensor

from math_functions import solve_quadratic, sqrt_zero
from testing import comparetensors


def test_sqrt_zero():
    """
    Test sqrt_zero on a tensor with some negative and positive numbers.
    """
    a = Tensor(((9.0, 1, 0), (-4, -1, 1e-6)))
    b = sqrt_zero(a)
    b_manual = Tensor(((3.0, 1, 0), (0, 0, 1e-3)))
    comparetensors(b, b_manual)

    a2 = -1.
    b2 = sqrt_zero(a2)
    b2_manual = 0
    comparetensors(b2, b2_manual)


def test_solve_quadratic():
    """
    Test solving 0 = (x-1)(2x+4) = 2x^2 + 2x - 4.
    Solutions: x=1 or x=-2
    """
    a = Tensor((2.,))
    b = Tensor((2.,))
    c = Tensor((-4.,))
    xsolutions = Tensor(solve_quadratic(a, b, c))
    comparetensors(xsolutions, Tensor((1, -2)))
