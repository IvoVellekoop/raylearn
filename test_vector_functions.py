"""Test vector functions."""

from torch import Tensor
import numpy as np

from testing import comparetensors
from vector_functions import rotate, unit


def test_rotate1():
    """Simple rotation test around z axis."""
    theta = np.pi / 4
    v = Tensor((2, 0, 0))
    u = Tensor((0, 0, 1))
    v_rot = rotate(v, u, theta)
    sqrt2 = float(np.sqrt(2))
    assert comparetensors(v_rot, sqrt2 * Tensor((1, 1, 0)))


def test_rotate2():
    """Simple rotation test around x=y axis."""
    theta = Tensor((np.pi,))
    v = Tensor(((4, 1, 6), (3, 2, 5)))
    u = unit(Tensor((1, 1, 0)))
    v_rot = rotate(v, u, theta)
    v_rot_test = Tensor(((1, 4, -6), (2, 3, -5)))
    assert comparetensors(v_rot, v_rot_test)


if __name__ == '__main__':
    """If file is executed as script, run the functions."""
    test_rotate1()
    test_rotate2()
