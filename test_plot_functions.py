"""Test plot functions."""

from torch import tensor
from testing import comparetensors
from ray_plane import Ray
from plot_functions import format_prefix


def test_format_prefix():
    # Test prefix formatting
    assert(format_prefix(3.1426, '.3f') == '3.143')
    assert(format_prefix(23e-6, '.1f') == '23.0µ')
    assert(format_prefix(-7e3, '4.0f') == '  -7k')
    assert(format_prefix(tensor((23e-9,), requires_grad=True), '.2f') == '23.00n')


