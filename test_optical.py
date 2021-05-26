"""Test optical."""

import torch
from torch import Tensor, tensor   ##### torch.Tensor does weird stuff when you give it a torch.Size
import numpy as np

from testing import comparetensors
from vector_functions import dot, unit, norm, cross, rejection
from ray_plane import Ray, Plane, CoordPlane
from optical import point_source, collimated_source, ideal_lens, snells


def test_sources():
    """
    Test point source and collimated source.

    Create a point source. Check position and directions.
    Create collimated source. Check direction and positions.
    """
    Nx = 2
    Ny = 4
    position = Tensor((1, 2, 3))
    x = Tensor((-3, 3, 0))
    y_prime = Tensor((1, 2, 5))
    y = rejection(y_prime, x)
    normal = unit(cross(x, y))
    sourceplane = CoordPlane(position, x, y)

    # Check coordplane vectors orthogonality
    assert comparetensors(dot(x, y), 0)
    assert comparetensors(dot(x, normal), 0)
    assert comparetensors(dot(y, normal), 0)

    # Point source
    intensity = torch.linspace(0.2, 1, Nx).view(1, Nx, 1)
    point = point_source(sourceplane, Nx, Ny, intensity=intensity)
    assert comparetensors(point.position_m, position)
    assert comparetensors(point.direction[Nx-1, Ny-1, :], unit(normal + x + y).view(3))
    assert comparetensors(point.direction[0, 0, :],       unit(normal - x - y).view(3))
    assert comparetensors(point.direction[0, Ny-1, :],    unit(normal - x + y).view(3))
    assert torch.all(tensor(point.direction.shape) == Tensor((Nx, Ny, 3)))

    # Collimated beam
    coll = collimated_source(sourceplane, Nx, Ny)
    assert comparetensors(coll.direction, normal)
    assert comparetensors(coll.position_m[Nx-1, Ny-1, :], (position + x + y).view(3))
    assert comparetensors(coll.position_m[0, 0, :],       (position - x - y).view(3))
    assert comparetensors(coll.position_m[0, Ny-1, :],    (position - x + y).view(3))
    assert torch.all(tensor(coll.position_m.shape) == Tensor((Nx, Ny, 3)))


def test_ideal_lens_point_source():
    """Test point source at focal plane through ideal lens."""
    Nx = 4
    Ny = 4

    # Define point source
    src_pos = Tensor((2,5,4))
    src_x = Tensor((1,6,9))
    src_y_prime = Tensor((4,2,0))
    src_y = unit(rejection(src_y_prime, src_x))
    src = point_source(CoordPlane(src_pos, src_x, src_y), Nx, Ny)

    # Define lens at distance f
    f = 3
    lens_dir = unit(Tensor((2,3,-5)))
    lens_pos = src_pos - f * lens_dir + rejection(Tensor((2, -1, 0.1)), lens_dir)
    lens = Plane(lens_pos, lens_dir)

    # Check output rays point source through lens
    outray = ideal_lens(src, lens, f)
    assert comparetensors(outray.position_m, src.intersect_plane(lens).position_m)
    assert comparetensors(outray.direction, unit(lens_pos - src_pos))


def test_ideal_lens_collimated_source():
    """Test collimated source through lens."""
    Nx = 1
    Ny = 5

    # Define collimated source
    src_pos = Tensor((-1,4,6))
    src_x = Tensor((3,4,2))
    src_y_prime = Tensor((-5,-6,-1))
    src_y = unit(rejection(src_y_prime, src_x))
    src_plane = CoordPlane(src_pos, src_x, src_y)
    src = collimated_source(src_plane, Nx, Ny)

    # Define lens
    f = 6
    lens_dir = unit(Tensor((8,1,-3)))
    lens_pos = Tensor((-2,-0.1,9))
    lens = Plane(lens_pos, lens_dir)

    # Define back focal plane
    focal_pos = lens_pos - f * lens_dir
    BFP = Plane(focal_pos, lens_dir)

    # Check output rays collimated source through lens
    on_lens_ray = ideal_lens(src, lens, f)
    focussed_ray = on_lens_ray.intersect_plane(BFP)
    chief_ray_at_BFP = Ray(lens_pos, src_plane.normal).intersect_plane(BFP)

    assert comparetensors(on_lens_ray.position_m, src.intersect_plane(lens).position_m)
    assert comparetensors(focussed_ray.position_m, chief_ray_at_BFP.position_m)


def test_ideal_lens_lens_law_positive():
    """Test lens law with point source through ideal positive lens."""
    Nx = 5
    Ny = 4

    # Define point source
    src_pos = Tensor((1,0,3))
    src_x = Tensor((-0.4,1,2))
    src_y_prime = Tensor((0,-0.9,4))
    src_y = unit(rejection(src_y_prime, src_x))
    src = point_source(CoordPlane(src_pos, src_x, src_y), Nx, Ny)

    # Define lens at distance s1
    f = 3
    s1 = 2.6 * f
    s2 = 1 / (1/f - 1/s1)
    lens_dir = unit(Tensor((2,1,-6)))
    lens_pos = src_pos - s1 * lens_dir + rejection(Tensor((0.3,1.2,-0.4)), lens_dir)
    lens = Plane(lens_pos, lens_dir)

    image_plane_pos = lens_pos - s2 * lens_dir + rejection(Tensor((-1,3,0.2)), lens_dir)
    image_plane = Plane(image_plane_pos, lens_dir)

    # Check whether focus is formed at distance s2
    focussed_ray = ideal_lens(src, lens, f).intersect_plane(image_plane)
    assert comparetensors(focussed_ray.position_m.std(dim=(0,1)), 0)


def test_ideal_lens_lens_law_negative():
    """Test lens law with point source through ideal negative lens."""
    Nx = 4
    Ny = 3

    # Define point source
    src_pos = Tensor((2.4,-1,-2))
    src_x = Tensor((-3,1,2))
    src_y_prime = Tensor((0,-4,-1))
    src_y = unit(rejection(src_y_prime, src_x))
    src = point_source(CoordPlane(src_pos, src_x, src_y), Nx, Ny)

    # Define lens at distance s1
    f = -42
    s1 = 0.7 * f
    s2 = 1 / (1/f - 1/s1)
    lens_dir = unit(Tensor((3,0,-7)))
    lens_pos = src_pos - s1 * lens_dir + rejection(Tensor((1.2,-1.8,0)), lens_dir)
    lens = Plane(lens_pos, lens_dir)

    image_plane_pos = lens_pos - s2 * lens_dir + rejection(Tensor((1,0.3,0.5)), lens_dir)
    image_plane = Plane(image_plane_pos, lens_dir)

    # Check whether focus is formed at distance s2
    focussed_ray = ideal_lens(src, lens, f).intersect_plane(image_plane)
    assert comparetensors(focussed_ray.position_m.std(dim=(0,1)), 0, 500)


def test_snells1():
    """
    Test Snell's law on a simple manually constructed case. For easyness,
    xz-plane is used, with normal = -z. Input angle = 45 degrees.
    """
    n_in = 1.5
    n_out = 1.2

    # Define input Ray
    ray_in_pos = Tensor((0,0,0))
    ray_in_dir = unit(Tensor((1,0,1)))
    ray_in = Ray(ray_in_pos, ray_in_dir, refractive_index=n_in)

    normal = Tensor((0,0,-1))
    ray_out = snells(ray_in, normal, n_out)

    # Since directional and normal vectors are unitary, ||DxN|| = sin(angle)
    # where x denotes cross product and |.| norm.
    sin_in = np.sin(np.pi/4)
    sin_out = n_in / n_out * sin_in
    assert ray_out.refractive_index == n_out
    assert comparetensors(sin_out, ray_out.direction[0])


def test_snells2():
    """
    Test Snell's law on a point source and surface normal in arbitrary
    direction. Sines are computed using cross products.
    """
    Nx = 4
    Ny = 4

    n_in = 1.3
    n_out = 1.7

    # Define point source
    src_pos = Tensor((2,42,3))
    src_x = Tensor((3,-0.2,0.3))
    src_y_prime = Tensor((1,4,0))
    src_y = unit(rejection(src_y_prime, src_x))
    src = point_source(CoordPlane(src_pos, src_x, src_y), Nx, Ny, refractive_index=n_in)

    normal = unit(Tensor((3,2,-9)))
    ray_out = snells(src, normal, n_out)

    # Since directional and normal vectors are unitary: ||DxN|| = sin(angle)
    sin_in = norm(cross(src.direction, normal))
    sin_out = norm(cross(ray_out.direction, normal))
    assert comparetensors(sin_out, n_in/n_out * sin_in)


if __name__ == '__main__':
    test_sources()
