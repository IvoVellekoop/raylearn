"""
Test math functions
"""

from torch import Tensor

from math_functions import solve_quadratic
from testing import comparetensors


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
