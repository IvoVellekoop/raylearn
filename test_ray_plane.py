"""Test ray_plane."""

import torch

from vector_functions import cartesian3d
from ray_plane import Plane, CoordPlane
from optical import point_source
from testing import comparetensors


torch.set_default_tensor_type('torch.DoubleTensor')


def test_backpropagation():
    """Send rays from a point source back and forth."""

    origin, x, y, z = cartesian3d()

    # Settings
    n = 1.5
    Nx = 3
    Ny = 5

    # Create source
    sourceplane = CoordPlane(origin, x, y)
    source = point_source(sourceplane, Nx, Ny, refractive_index=n)

    # Create plane for intersection
    interplane = Plane(origin + z, -z)

    # Forward and backward propagation
    ray1 = source.intersect_plane(interplane)
    ray2 = ray1.intersect_plane(sourceplane)

    assert comparetensors(ray2.pathlength_m, source.pathlength_m)
