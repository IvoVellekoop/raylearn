"""Test plot functions."""

import pytest
from torch import tensor, ones
from testing import comparetensors
from ray_plane import Ray
from vector_functions import unit
from plot_functions import format_prefix, ray_positions


class DummyAxis:
    """
    Dummy axis for capturing the input to the plot function.
    """
    def plot(self, *args):
        self.args = args


def test_format_prefix():
    """
    Test prefix formatting.
    """
    assert(format_prefix(3.1426, '.3f') == '3.143')
    assert(format_prefix(23e-6, '.1f') == '23.0µ')
    assert(format_prefix(-7e3, '4.0f') == '  -7k')
    assert(format_prefix(7.1583, '.3g') == '7.16')
    assert(format_prefix(.071983, '.3g') == '72m')
    assert(format_prefix(.71983, '.3g') == '720m')
    assert(format_prefix(7.1983, '.3g') == '7.2')
    assert(format_prefix(71.983, '.3g') == '72')
    assert(format_prefix(7198.3, '.3g') == '7.2k')
    assert(format_prefix(tensor((23e-9,), requires_grad=True), '.2f') == '23.00n')


def test_ray_positions():
    """
    Test ray positions. This function extracts ray positions from a ray list
    and expands them so the shapes are compatible (also for plotting).
    """
    # Create positions and direction
    N1 = 4
    N2 = 2
    N3 = 5
    pos0 = 1 * ones((N3,))
    pos1 = 2 * ones((N1, 1, N3))
    pos2 = 3 * ones((1, N2, N3))
    dir0 = unit(ones((3,)))

    # Create Rays
    raylist = []
    raylist.append(Ray(pos0, dir0))
    raylist.append(Ray(pos1, dir0))
    raylist.append(Ray(pos2, dir0))

    positions = ray_positions(raylist)
    shape0 = tensor(positions[0].shape)
    shape1 = tensor(positions[1].shape)
    shape2 = tensor(positions[2].shape)

    assert(comparetensors(shape0, tensor((N1, N2, N3))))
    assert(comparetensors(shape0, shape1))
    assert(comparetensors(shape0, shape2))
