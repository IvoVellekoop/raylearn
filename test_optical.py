"""Test optical."""

import torch
from torch import Tensor, tensor, stack, meshgrid
import numpy as np

from testing import comparetensors
from vector_functions import dot, unit, norm, cross, rejection, rotate,\
                             reflection, components, cartesian3d
from ray_plane import Ray, Plane, CoordPlane
from optical import point_source, collimated_source, ideal_lens, snells,\
                    mirror, galvo_mirror, slm_segment


###############
import matplotlib.pyplot as plt
from plot_functions import plot_plane, plot_lens, plot_rays
###############


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
    src_pos = Tensor((2,1,4))
    src_x = Tensor((0.5,0.3,0.2))
    src_y_prime = Tensor((0,3,1))
    src_y = unit(rejection(src_y_prime, src_x))
    src = point_source(CoordPlane(src_pos, src_x, src_y), Nx, Ny)

    # Define lens at distance f
    f = 3
    lens_dir = unit(Tensor((2,3,-5)))
    lens_pos = src_pos - f * lens_dir + rejection(Tensor((0, -1, 0.1)), lens_dir)
    lens = Plane(lens_pos, lens_dir)

    # Check output rays point source through lens
    outray = ideal_lens(src, lens, f)

    assert comparetensors(outray.position_m, src.intersect_plane(lens).position_m)
    assert comparetensors(outray.direction, unit(lens_pos - src_pos))


def test_ideal_lens_collimated_source():
    """Test collimated source through lens."""
    Nx = 3
    Ny = 5

    # Define collimated source
    src_pos = Tensor((-3,2,5))
    src_x = Tensor((2,3,1))
    src_y_prime = Tensor((-5,-6,-1))
    src_y = 3*unit(rejection(src_y_prime, src_x))
    src_plane = CoordPlane(src_pos, src_x, src_y)
    src = collimated_source(src_plane, Nx, Ny)

    # Define lens
    f = 6
    lens_dir = rotate(-src_plane.normal, unit(src_x), 0.15)
    lens_pos = Tensor((-2,-0.1,9))
    lens = Plane(lens_pos, lens_dir)

    # Define back focal plane
    focal_pos = lens_pos - f * lens_dir
    BFP = Plane(focal_pos, lens_dir)

    # Check output rays collimated source through lens
    on_lens_ray = ideal_lens(src, lens, f)
    focussed_ray = on_lens_ray.intersect_plane(BFP)
    chief_ray_at_BFP = Ray(lens_pos, src_plane.normal).intersect_plane(BFP)

    breakpoint()

    #########################
    fig = plt.figure(figsize=(7, 4))
    fig.dpi = 144
    ax1 = plt.gca()

    # Plot lenses and planes
    viewplane = CoordPlane(Tensor((0,0,0)), src_plane.normal, unit(src_y))
    plot_lens(ax1, viewplane, lens, f, 6, 'Lens')
    plot_plane(ax1, viewplane, BFP, 3)

    # Plot rays
    plot_rays(ax1, viewplane, [src, on_lens_ray, endray])

    plt.show()
    #########################

    assert comparetensors(on_lens_ray.position_m, src.intersect_plane(lens).position_m)
    assert comparetensors(focussed_ray.position_m, chief_ray_at_BFP.position_m)

    # Light in focus should be in phase
    print(focussed_ray.pathlength_m)
    assert comparetensors(focussed_ray.pathlength_m, focussed_ray.pathlength_m[0,0,0])


def test_ideal_lens_lens_law_positive():
    """Test lens law with point source through ideal positive lens."""
    Nx = 5
    Ny = 4

    # Define point source
    src_pos = Tensor((1,0,3))
    src_x = Tensor((1.6,1,0))
    src_y_prime = Tensor((0, 9, 4))
    src_y = unit(rejection(src_y_prime, src_x))
    src_plane = CoordPlane(src_pos, src_x, src_y)
    src = point_source(src_plane, Nx, Ny)

    # Define lens at distance s1
    f = 3
    s1 = 2.6 * f
    s2 = 1 / (1/f - 1/s1)
    # lens_dir = unit(Tensor((2,1,-6)))
    lens_dir = -src_plane.normal
    lens_pos = src_pos - s1 * lens_dir + 0*rejection(Tensor((0.3,1.2,-0.4)), lens_dir)
    lens = Plane(lens_pos, lens_dir)

    image_plane_pos = lens_pos - s2 * lens_dir + rejection(Tensor((-1,3,0.2)), lens_dir)
    image_plane = Plane(image_plane_pos, lens_dir)

    # Check whether focus is formed at distance s2
    focussed_ray = ideal_lens(src, lens, f).intersect_plane(image_plane)
    assert comparetensors(focussed_ray.position_m.std(dim=(0,1)), 0)
    assert comparetensors(focussed_ray.pathlength_m, focussed_ray.pathlength_m[0,0])


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
    src_pos = tensor((2., 42., 3.))
    src_x = tensor((3, -0.2, 0.3))
    src_y_prime = tensor((1., 4., 0.))
    src_y = unit(rejection(src_y_prime, src_x))
    src_plane = CoordPlane(src_pos, src_x, src_y)
    src = point_source(src_plane, Nx, Ny, refractive_index=n_in)

    normal = unit(tensor((3., 2., -9.)))
    ray_out = snells(src, normal, n_out)

    # Since directional and normal vectors are unitary: ||DxN|| = sin(angle)
    sin_in = norm(cross(src.direction, normal))
    sin_out = norm(cross(ray_out.direction, normal))
    assert comparetensors(sin_out, n_in/n_out * sin_in)


def test_mirror1():
    """
    Test mirror with single manual Ray.
    """
    # Construct Mirror Plane
    theta = 0.42
    origin = tensor((0., 0., 0.))
    x = tensor((1., 0., 0.))
    z = tensor((0., 0., 1.))
    mirror_normal = rotate(x, z, theta)
    mirror_plane = Plane(origin, mirror_normal)

    # Construct Rays
    ray_in = Ray(x, -x)
    ray_reflect = mirror(ray_in, mirror_plane)
    ray_dir_manual = tensor((np.cos(2*theta), np.sin(2*theta), 0))

    assert comparetensors(ray_dir_manual, ray_reflect.direction)
    assert comparetensors(origin, ray_reflect.position_m)


def test_mirror2():
    """
    Test mirror with mirror version of point source.

    Note: reflecting along z == rotating around y + reflecting along x
    """
    Nx = 5
    Ny = 3

    # Construct randomly oriented coordinate system
    origin = tensor((0., 0., 0.))
    x = unit(torch.randn(3))
    y = unit(rejection(torch.randn(3), x))
    z = cross(x, y)

    # Mirror plane normal = -z
    mirror_plane = Plane(origin, z)

    # Define point source
    src_plane = CoordPlane(-2*z, x, y)
    src = point_source(src_plane, Nx, Ny)

    # Reflected rays
    reflected_rays = mirror(src, mirror_plane)

    # Rotated point source around y, then reflect along x
    mirror_src_plane = CoordPlane(2*z, -x, y)
    mirror_src = point_source(mirror_src_plane, Nx, Ny)
    refl_mirror_dirs = reflection(mirror_src.direction, x)

    assert comparetensors(reflected_rays.direction, refl_mirror_dirs)


def test_galvo_mirror1():
    """
    Test galvo mirror for a simple case.

    Rotate the mirror in steps of 22.5 degrees and check against
    manual computation.
    """
    # Construct randomly oriented coordinate system
    origin = torch.randn(3)
    x = unit(torch.randn(3))
    y = unit(rejection(torch.randn(3), x))
    z = cross(x, y)

    # Can be uncommented for easier troubleshooting:
    # x = tensor((1.,0,0))
    # y = tensor((0.,1,0))
    # z = tensor((0.,0,1))

    # Galvo Plane with y+z-diagonal as normal and input Ray coming in from y
    galvo_plane = CoordPlane(origin, x, unit(y-z))

    assert comparetensors(galvo_plane.normal, unit(z+y))

    ray_in = Ray(origin + y, -y)

    # Rotate with steps of 22.5 degrees around x axis
    # -> reflection direction of ray should rotate in steps of 45 degrees
    rotations = np.pi/8 * tensor(((2., 0), (1, 0), (0, 0), (-1, 0), (-2, 0)))
    reflected_rays = galvo_mirror(ray_in, galvo_plane, rotations)

    # Manually compute direction unit vectors
    sq2 = np.sqrt(0.5, dtype=np.float32)
    x_dir_man = tensor((0., 0., 0., 0., 0.)).view(5, 1) * x
    y_dir_man = tensor((-1, -sq2, 0, sq2, 1)).view(5, 1) * y
    z_dir_man = tensor((0., sq2, 1., sq2, 0)).view(5, 1) * z
    direction_man = x_dir_man + y_dir_man + z_dir_man

    assert comparetensors(direction_man, reflected_rays.direction)


def test_galvo_mirror2():
    """
    Test galvo mirror with 2 rotation axes.

    Galvo Plane axes are -x&z. Input Ray has direction -z.
    """
    # Construct randomly oriented coordinate system
    x = unit(torch.randn(3))
    y = unit(rejection(torch.randn(3), x))
    z = cross(x, y)

    # Can be uncommented for easier troubleshooting:
    # x = tensor((1.,0,0))
    # y = tensor((0.,1,0))
    # z = tensor((0.,0,1))

    # Galvo Plane and input Ray
    galvo_plane = CoordPlane(torch.randn(3), -x, z)
    ray_in = Ray(torch.randn(3), -z)

    # Reflect rays off rotated galvo mirrors
    N = 7
    rot_lin = np.pi/2 * torch.linspace(-1, 1, N)
    rotations = stack(meshgrid(rot_lin, rot_lin*0), dim=-1)
    reflected_rays = galvo_mirror(ray_in, galvo_plane, rotations)
    shape = torch.tensor(reflected_rays.direction.shape)

    assert comparetensors(shape, torch.tensor((N, N, 3)))

    # Manually compute directions for comparison
    # Note! The galvo directions here are global -x&z direction
    rot_galvox, rot_galvoy = components(rotations)

    # Manual components for this particular situation
    x_dir_man = x *  torch.sin(2*rot_galvox) * torch.sin(rot_galvoy)
    y_dir_man = y * -torch.sin(2*rot_galvox) * torch.cos(rot_galvoy)
    z_dir_man = z * -torch.cos(2*rot_galvox)
    direction_man = x_dir_man + y_dir_man + z_dir_man

    assert comparetensors(direction_man, reflected_rays.direction)


def test_galvo_mirror3():
    """
    Test Galvo Mirror reflected Ray position.
    """
    # Construct randomly oriented coordinate system
    x = unit(torch.randn(3))
    y = unit(rejection(torch.randn(3), x))
    z = cross(x, y)

    # Planes and Ray
    galvo_plane = CoordPlane(torch.randn(3), x, y)
    ray_in = Ray(torch.randn(3), unit(torch.randn(3)))
    rot_mir_plane_man = Plane(galvo_plane.position_m, unit(z-y))

    # Rotate 45 degrees around x
    rot_x = float(np.pi/4)
    rotations = torch.tensor((rot_x, 0))

    # Compute positions
    position_at_galvo = galvo_mirror(ray_in, galvo_plane, rotations).position_m
    position_man = ray_in.intersect_plane(rot_mir_plane_man).position_m

    assert comparetensors(position_man, position_at_galvo)


def test_slm_segment():
    """
    Test SLM segment.
    """
    # Construct randomly oriented coordinate system
    x = unit(torch.randn(3))
    y = unit(rejection(torch.randn(3), x))
    z = cross(x, y)

    # Construct SLM stuff and Ray
    n_ray = 1.42
    pathlength_m = 7
    ray_in = Ray(torch.randn(3), unit(torch.randn(3)),
                        refractive_index=n_ray, pathlength_m=pathlength_m)
    slm_plane = CoordPlane(torch.randn(3), 0.1*x, 0.2*y)
    slm_coords = torch.randn((4,2))

    # Compute output Ray and manual positions
    ray_out = slm_segment(ray_in, slm_plane, slm_coords)
    x_slm, y_slm = components(slm_coords)        # Split vector dimensions
    position_man = slm_plane.position_m + x_slm * slm_plane.x + y_slm * slm_plane.y

    assert comparetensors(ray_out.direction, ray_in.direction)
    assert ray_out.refractive_index == n_ray
    assert ray_out.pathlength_m == pathlength_m
    assert comparetensors(ray_out.position_m, position_man)


def test_pathlength():
    """
    Test path length for a collimated beam in free space and glass
    """
    origin, x, y, z = cartesian3d()
    
    # medium consists of two slices with thickness 0.1m
    n1 = 1.0
    n2 = 1.5

    d = 0.1
    
    sourceplane = CoordPlane(origin, x, y)
    interfaceplane = CoordPlane(origin + d*z, x, y)
    outputplane = CoordPlane(origin + 2*d*z, x, y)
    
    ray_in = collimated_source(sourceplane,1,1, refractive_index=n1)
    
    # calculate new position and direction/IOR
    ray_out = ray_in.intersect_plane(interfaceplane)
    ray_out = snells(ray_out, interfaceplane.normal, n2)
    
    ray_out = ray_out.intersect_plane(outputplane)

    gt_pathlength = d*n1 + d*n2
    assert comparetensors(ray_out.pathlength_m, gt_pathlength) 

def test_pathlength2():
    """
    Test path length for a lens system
    """
    origin, x, y, z = cartesian3d()

    beamwidth = 10e-3
    src_plane = CoordPlane(origin + y*beamwidth, x*beamwidth, y*beamwidth*np.cos(-0.1) - z*beamwidth*np.sin(-0.1))
    f = 100e-3
    lens_plane = Plane(origin + f*z, -z-0.1*y - 0.1*x)
    cam_plane = Plane(lens_plane.position_m - f*lens_plane.normal, lens_plane.normal)

    rays = [collimated_source(src_plane, 3, 3)]
    rays.append(ideal_lens(rays[-1], lens_plane, f))
    rays.append(rays[-1].intersect_plane(cam_plane))

    assert comparetensors(rays[-1].pathlength_m, rays[-1].pathlength_m[0][0])
