"""Test vector functions."""

import torch
import numpy as np

from testing import comparetensors
from vector_functions import rotate, vector, unit


def test_rotate1():
    """Simple rotation test around z axis."""
    sz = (3, 1, 1)
    theta = np.pi / 4
    v = vector((2, 0, 0), sz)
    u = vector((0, 0, 1), sz)
    v_rot = rotate(v, u, theta)
    sqrt2 = float(np.sqrt(2))
    assert comparetensors(v_rot, sqrt2 * vector((1, 1, 0), sz))


def test_rotate2():
    """Simple rotation test around x=y axis."""
    sz = (3, 2)
    theta = torch.tensor(np.pi)
    v = torch.tensor(((4, 3), (1, 2), (6, 5)))
    u = unit(vector((1, 1, 0), sz))
    v_rot = rotate(v, u, theta)
    v_rot_test = torch.tensor(((1, 2), (4, 3), (-6, -5)))
    assert comparetensors(v_rot, v_rot_test)
