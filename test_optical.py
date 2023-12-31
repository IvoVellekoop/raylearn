"""Test optical."""

import torch
from torch import Tensor, tensor, stack, meshgrid
import numpy as np

from testing import comparetensors
from vector_functions import dot, unit, norm, cross, rejection, rotate,\
                             reflection, components, cartesian3d
from math_functions import sign
from ray_plane import Ray, Plane, CoordPlane
from optical import point_source, collimated_source, thin_lens, abbe_lens, snells, flat_interface, \
                    propagate_to_cylinder, cylinder_interface, mirror, galvo_mirror, slm_segment


torch.set_default_tensor_type('torch.DoubleTensor')


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


def test_thin_lens_point_source():
    """Test point source at focal plane through ideal lens."""
    Nx = 4
    Ny = 4

    # Define point source
    src_pos = Tensor((2, 1, 4))
    src_x = Tensor((0.5, 0.3, 0.2))
    src_y_prime = Tensor((0, 3, 1))
    src_y = unit(rejection(src_y_prime, src_x))
    src = point_source(CoordPlane(src_pos, src_x, src_y), Nx, Ny)

    # Define lens at distance f
    f = 3
    lens_dir = unit(Tensor((2, 3, -5)))
    lens_pos = src_pos - f * lens_dir + rejection(Tensor((0, -1, 0.1)), lens_dir)
    lens = Plane(lens_pos, lens_dir)

    # Check output rays point source through lens
    outray = thin_lens(src, lens, f)

    assert comparetensors(outray.position_m, src.intersect_plane(lens).position_m)
    assert comparetensors(outray.direction, unit(lens_pos - src_pos))


def test_thin_lens_collimated_source():
    """Test collimated source through lens."""
    Nx = 3
    Ny = 5

    # Define collimated source
    src_pos = Tensor((-3, 2,5))
    src_x = Tensor((2, 3,1))
    src_y_prime = Tensor((-5, -6, -1))
    src_y = 3*unit(rejection(src_y_prime, src_x))
    src_plane = CoordPlane(src_pos, src_x, src_y)
    src = collimated_source(src_plane, Nx, Ny)

    # Define lens
    f = 6
    lens_dir = rotate(-src_plane.normal, unit(src_x), 0.15)
    lens_pos = Tensor((-2, -0.1, 9))
    lens = Plane(lens_pos, lens_dir)

    # Define back focal plane
    focal_pos = lens_pos - f * lens_dir
    BFP = Plane(focal_pos, lens_dir)

    # Check output rays collimated source through lens
    on_lens_ray = thin_lens(src, lens, f)
    focussed_ray = on_lens_ray.intersect_plane(BFP)
    chief_ray_at_BFP = Ray(lens_pos, src_plane.normal).intersect_plane(BFP)

    assert comparetensors(on_lens_ray.position_m, src.intersect_plane(lens).position_m)
    assert comparetensors(focussed_ray.position_m, chief_ray_at_BFP.position_m)


def test_thin_lens_lens_law_positive():
    """Test lens law with point source through ideal positive lens."""
    Nx = 5
    Ny = 4

    # Define point source
    src_pos = Tensor((1, 0, 3))
    src_x = Tensor((1.6, 1, 0))
    src_y_prime = Tensor((0, 9, 4))
    src_y = unit(rejection(src_y_prime, src_x))
    src_plane = CoordPlane(src_pos, src_x, src_y)
    src = point_source(src_plane, Nx, Ny)

    # Define lens at distance s1
    f = 3
    s1 = 2.6 * f
    s2 = 1 / (1/f - 1/s1)
    lens_dir = -src_plane.normal

    # Define lens such that point source is at focal plane, but not at focal point
    lens_pos = src_pos - s1 * lens_dir + 0.1*rejection(Tensor((0.3, 1.2, -0.4)), lens_dir)
    lens = Plane(lens_pos, lens_dir)

    # Define image plane at back focal plane, with center 
    image_plane_pos = lens_pos - s2 * lens_dir + rejection(Tensor((-1, 3, 0.2)), lens_dir)
    image_plane = Plane(image_plane_pos, lens_dir)

    # Check whether focus is formed at distance s2
    focussed_ray = thin_lens(src, lens, f).intersect_plane(image_plane)
    assert comparetensors(focussed_ray.position_m.std(dim=(0, 1)), 0)


def test_thin_lens_lens_law_negative():
    """Test lens law with point source through ideal negative lens."""
    Nx = 3
    Ny = 4

    origin, x, y, z = cartesian3d()

    # Define point source
    src_plane = CoordPlane(origin, 0.1*x, 0.1*rotate(y, x, 0.3))
    src = point_source(src_plane, Nx, Ny)

    # Define lens at distance s1
    f = -4.2
    s1 = -2.7 * f
    s2 = 1 / (1/s1 - 1/f)
    lens_dir = unit(Tensor((-0.1, 0.5, -7)))
    lens_pos = origin - s1 * lens_dir + rejection(Tensor((2.2, -2.8, 0)), lens_dir)
    lens_plane = Plane(lens_pos, lens_dir)

    # Back focal plane
    BFP_pos = lens_pos + f * lens_dir
    BFP = Plane(BFP_pos, lens_dir)

    # Image plane
    image_plane_pos = lens_pos + s2 * lens_dir + rejection(Tensor((2, 0.5, 0.5)), lens_dir)
    image_plane = Plane(image_plane_pos, lens_dir)

    # Check whether focus is formed at distance s2
    ray_at_lens = thin_lens(src, lens_plane, f)
    ray_bfp = ray_at_lens.intersect_plane(BFP)
    focussed_ray = ray_bfp.intersect_plane(image_plane)

    assert comparetensors(focussed_ray.position_m.std(dim=(0, 1)), 0)


def test_snells1():
    """
    Test Snell's law on a simple manually constructed case. For easyness,
    xz-plane is used, with normal = -z. Input angle = 45 degrees.
    """
    n_in = 1.5
    n_out = 1.2

    origin, x, y, z = cartesian3d()

    # Define input Ray
    ray_in_pos = origin
    ray_in_dir = unit(x + z)
    ray_in = Ray(ray_in_pos, ray_in_dir, refractive_index=n_in)

    # Output Ray
    normal = -z
    ray_out = snells(ray_in, normal, n_out)

    # Snell's law check
    sin_in = np.sin(np.pi/4)
    sin_out = n_in / n_out * sin_in
    angle_out_rad = np.arcsin(sin_out)

    assert ray_out.refractive_index == n_out

    # Plane is aligned with coordinate system, so we can simply take the vector components
    assert comparetensors(sin_out, ray_out.direction[0])
    assert comparetensors(np.cos(angle_out_rad), ray_out.direction[2])
    assert sign(ray_out.direction[2]) == sign(ray_in.direction[2])


def test_snells2():
    """
    Test Snell's law on a simple manually constructed case with flipped normal. For easyness,
    xz-plane is used, with normal = z. (Opposite normal as test_snells1).
    The x- and z-components are used for comparison.
    """
    n_in = 1.6
    n_out = 1.3
    angle_deg = 35

    origin, x, y, z = cartesian3d()

    # Define input Ray
    angle_rad = angle_deg * np.pi/180
    ray_in_pos = origin
    ray_in_dir = rotate(z, y, angle_rad)
    ray_in = Ray(ray_in_pos, ray_in_dir, refractive_index=n_in)

    # Output Ray
    normal = z
    ray_out = snells(ray_in, normal, n_out)

    # Snell's law check
    sin_in = np.sin(angle_rad)
    sin_out = n_in / n_out * sin_in
    angle_out_rad = np.arcsin(sin_out)

    assert ray_out.refractive_index == n_out

    # Plane is aligned with coordinate system, so we can simply take the vector components
    assert comparetensors(sin_out, ray_out.direction[0])
    assert comparetensors(np.cos(angle_out_rad), ray_out.direction[2])
    assert sign(ray_out.direction[2]) == sign(ray_in.direction[2])


def test_snells3():
    """
    Test returning of nan when total internal reflection occurs.
    """
    n_in = 2
    n_out = 1
    angle_deg = 50

    origin, x, y, z = cartesian3d()

    # Define input Ray
    angle_rad = angle_deg * np.pi/180
    ray_in_pos = origin
    ray_in_dir = rotate(z, y, angle_rad)
    ray_in = Ray(ray_in_pos, ray_in_dir, refractive_index=n_in)

    # output Ray
    normal = z
    ray_out = snells(ray_in, normal, n_out)

    sin_in = np.sin(angle_rad)
    sin_out = n_in / n_out * sin_in

    assert sin_out > 1
    assert torch.all(torch.isnan(ray_out.direction[0]))


def test_snells4():
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
    # where x denotes cross product and ||.|| norm.
    sin_in = norm(cross(src.direction, normal))
    sin_out = norm(cross(ray_out.direction, normal))
    assert comparetensors(sin_out, n_in/n_out * sin_in)


def test_snells_pathlength():
    """
    Test pathlength with forward and backward propagation through refractive interface. For
    easyness, xz-plane is used, with normal = -z.
    """
    n_in = 1.2
    n_out = 1.8
    angle_deg = 33

    origin, x, y, z = cartesian3d()

    # Define input Ray
    angle_in_rad = angle_deg * np.pi/180
    zdistance_in = 1.1
    start_position = origin - zdistance_in*z
    ray_in_dir = rotate(z, x, angle_in_rad)
    rays = [Ray(start_position, ray_in_dir, refractive_index=n_in)]

    # Define interface and start & end planes
    start_plane = Plane(start_position, -z)
    interface = Plane(origin, -z)
    zdistance_out = 0.8
    end_plane = Plane(origin + zdistance_out*z, -z)

    # Forward ray tracing
    rays += [rays[-1].intersect_plane(interface)]
    rays += [snells(rays[-1], interface.normal, n_out)]
    rays += [rays[-1].intersect_plane(end_plane)]

    # Check pathlength at end plane
    angle_out_rad = np.arcsin(np.sin(angle_in_rad) * n_in / n_out)      # Snell's law
    pathlength = n_in * zdistance_in / np.cos(angle_in_rad) \
        + n_out * zdistance_out / np.cos(angle_out_rad)
    assert comparetensors(pathlength, rays[-1].pathlength_m)

    # Backward ray tracing
    rays += [rays[-1].intersect_plane(interface)]
    rays += [snells(rays[-1], interface.normal, n_in)]
    rays += [rays[-1].intersect_plane(start_plane)]

    assert comparetensors(rays[0].position_m, rays[-1].position_m)
    assert comparetensors(rays[0].direction, rays[-1].direction)
    assert comparetensors(rays[0].pathlength_m, rays[-1].pathlength_m)


def test_propagate_to_cylinder():
    """
    Test propagation of point sources to cylindrical surface.
    """
    origin, x, y, z = cartesian3d()

    # Point sources
    Nx = 4
    Ny = 4
    tan_opening_angle1 = 0.25
    tan_opening_angle2 = 0.6
    tan_opening_angle3 = 0.3
    source_planes = []
    source_planes += [CoordPlane(1.5*z, tan_opening_angle1 * unit(x-z), tan_opening_angle1 * y)]
    source_planes += [CoordPlane(3*z, tan_opening_angle2 * unit(x-z), tan_opening_angle2 * y)]
    source_planes += [CoordPlane(3*z+y, tan_opening_angle3 * -x, tan_opening_angle3 * z)]
    propagation_sign = [1, 1, -1]

    # Cylinder
    cylinder_radius_m = 0.4
    cylinder_plane = CoordPlane(2*x+0.1*y+z, unit(x+z), y)

    raylists = [[], [], []]

    for i in range(3):
        # Ray trace to cylinder
        source_ray = point_source(source_planes[i], Nx, Ny)
        ray_at_cylinder = propagate_to_cylinder(source_ray, cylinder_plane, cylinder_radius_m,
                                                propagation_sign[i])

        raylists[i] += [source_ray]
        raylists[i] += [ray_at_cylinder]

        # Compute distance to cylinder axis
        C = cylinder_plane.position_m
        N = cylinder_plane.normal
        Q = ray_at_cylinder.position_m
        distance_to_cyl_axis = norm(rejection(Q-C, N))
        assert comparetensors(cylinder_radius_m, distance_to_cyl_axis)

    # # Uncomment for debugging
    # ### Plot ###
    # from matplotlib import pyplot as plt
    # from plot_functions import plot_rays, plot_cylinder
    # plt.figure()
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.title('propagate point sources to cylinder')
    # plot_text = ['1. outside', '2. inside', '3. backpropagation']
    # for i in range(3):
    #     plot_rays(ax, raylists[i], viewplane=cylinder_plane, plotkwargs={'color': f'C{i}'})
    #     xtext, ytext = components(cylinder_plane.transform_points(source_planes[i].position_m))
    #     plt.text(xtext, ytext, plot_text[i], backgroundcolor=(0.8, 0.8, 0.8, 0.5), ha='center')

    # # # Draw a circle
    # plt.plot(0, 0, '+k')
    # plot_cylinder(ax, cylinder_plane, cylinder_radius_m, 0, 0,
    #               num_verts_circle=120, viewplane=cylinder_plane)
    # plt.xlabel('cylinder plane x')
    # plt.ylabel('cylinder plane y')

    # plt.show()
    # ############


def test_cylinder_refraction():
    """
    Test refraction through a cylinder, aiming at the cylinder axis. This should be the same as
    refraction through a slab of the same thickness and refractive index.
    """
    origin, x, y, z = cartesian3d()

    # Point source ray elements
    Nx = 5
    Ny = 1

    # Parameters
    n_medium = 1.1
    n_cyl = 1.7
    radius_m = 0.55
    tan_opening_angle = 0.4

    # Planes
    source_plane = CoordPlane(z, -x, y)
    cylinder_plane = CoordPlane(origin, y, z)
    cam_plane = CoordPlane(-z, x, y)
    slab_front_plane = Plane(radius_m*z, -z)
    slab_back_plane = Plane(-radius_m*z, -z)

    # Ray tracing through cylinder
    rays_cyl = [point_source(source_plane, Nx, Ny, refractive_index=n_medium)]
    rays_cyl += [cylinder_interface(rays_cyl[-1], cylinder_plane, radius_m, n_cyl)]
    rays_cyl += [cylinder_interface(rays_cyl[-1], cylinder_plane, radius_m, n_medium)]
    rays_cyl += [rays_cyl[-1].intersect_plane(cam_plane)]

    # Ray tracing through slab
    rays_slab = rays_cyl[0:1]
    rays_slab += [flat_interface(rays_slab[-1], slab_front_plane, n_cyl)]
    rays_slab += [flat_interface(rays_slab[-1], slab_back_plane, n_medium)]
    rays_slab += [rays_cyl[-1].intersect_plane(cam_plane)]

    # # Uncomment for debugging
    # ### Plot ###
    # from matplotlib import pyplot as plt
    # from plot_functions import plot_plane, plot_rays
    # plt.figure()
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.title('propagate point source through slab and cylinder')
    # viewplane = CoordPlane(origin, x, z)
    # plot_plane(ax, slab_front_plane, scale=1.3, viewplane=viewplane)
    # plot_plane(ax, slab_back_plane, scale=1.3, viewplane=viewplane)
    # plot_rays(ax, rays_slab, viewplane=viewplane)

    # plt.xlabel('x')
    # plt.ylabel('z')

    # plt.show()
    # ############

    # Assertion
    assert comparetensors(rays_cyl[-1].position_m, rays_slab[-1].position_m)
    assert comparetensors(rays_cyl[-1].direction, rays_slab[-1].direction)
    assert comparetensors(rays_cyl[-1].pathlength_m, rays_slab[-1].pathlength_m)


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
    sq2 = torch.sqrt(Tensor((0.5,)))
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
    rotations = stack(meshgrid(rot_lin, rot_lin*0, indexing='xy'), dim=-1)
    reflected_rays = galvo_mirror(ray_in, galvo_plane, rotations)
    shape = torch.tensor(reflected_rays.direction.shape)

    assert comparetensors(shape, torch.tensor((N, N, 3)))

    # Manually compute directions for comparison
    # Note! The galvo directions here are global -x&z direction
    rot_galvox, rot_galvoy = components(rotations)

    # Manual components for this particular situation
    x_dir_man = x * torch.sin(2*rot_galvox) * torch.sin(rot_galvoy)
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

    # Construct SLM stuff and Ray
    n_ray = 1.42
    pathlength_m = 7
    ray_in = Ray(torch.randn(3), unit(torch.randn(3)), refractive_index=n_ray,
                 pathlength_m=pathlength_m)
    slm_plane = CoordPlane(torch.randn(3), 0.1*x, 0.2*y)
    slm_coords = torch.randn((4, 2))

    # Compute output Ray and manual positions
    ray_out = slm_segment(ray_in, slm_plane, slm_coords)
    x_slm, y_slm = components(slm_coords)        # Split vector dimensions
    position_man = slm_plane.position_m + x_slm * slm_plane.x + y_slm * slm_plane.y

    assert comparetensors(ray_out.direction, ray_in.direction)
    assert ray_out.refractive_index == n_ray
    assert ray_out.pathlength_m == pathlength_m
    assert comparetensors(ray_out.position_m, position_man)


def test_pathlength1():
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

    ray_in = collimated_source(sourceplane, 1, 1, refractive_index=n1)

    # calculate new position and direction/IOR
    ray1 = ray_in.intersect_plane(interfaceplane)
    ray2 = snells(ray1, interfaceplane.normal, n2)

    ray3 = ray2.intersect_plane(outputplane)

    gt_pathlength = d*n1 + d*n2
    assert comparetensors(ray3.pathlength_m, gt_pathlength)


def test_pathlength2():
    """
    Test path length for a collimated source and abbe sine lens
    """
    origin, x, y, z = cartesian3d()

    beamwidth = 1
    f = 2.5
    src_plane = CoordPlane(origin, x*beamwidth, beamwidth * rotate(y, x, 0.2))
    lens_plane = Plane(origin + 0.5*f*z, unit(-z-0.1*y - 0.1*x))
    cam_plane = Plane(lens_plane.position_m - f*lens_plane.normal, lens_plane.normal)

    rays = [collimated_source(src_plane, 3, 3)]
    rays += abbe_lens(rays[-1], lens_plane, f)
    rays += [rays[-1].intersect_plane(cam_plane)]

    assert comparetensors(rays[-1].pathlength_m.std(), 0)
    assert comparetensors(rays[-1].position_m.std(dim=(-2, -3)), 0)

    rays += abbe_lens(rays[-1], lens_plane, f)
    rays += [rays[-1].intersect_plane(src_plane)]

    # # Uncomment for debugging
    # ### Plot ###
    # from matplotlib import pyplot as plt
    # from plot_functions import plot_plane, plot_lens, plot_rays
    # plt.figure()
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.title('test_pathlength2')
    # plot_plane(ax, src_plane, text1='src')
    # plot_lens(ax, lens_plane, f, scale=2*beamwidth)
    # plot_plane(ax, cam_plane, scale=beamwidth, text1='cam')
    # plot_rays(ax, rays)
    # plt.show()
    # ############

    assert comparetensors(rays[-1].pathlength_m, 0)


def test_pathlength3():
    """
    Test pathlength for point source and abbe sine lens
    """
    origin, x, y, z = cartesian3d()

    # Source
    Nx = 9
    Ny = 9
    tan_angle = 0.4
    source_pos = origin + 0.1*y
    source_plane = CoordPlane(source_pos, tan_angle*x, tan_angle*y)
    source = point_source(source_plane, Nx, Ny)
    rays = [source]

    # Lens
    f = 0.23
    end_plane_to_lens = 0.4
    lens_pos = origin + f*source_plane.normal
    lens_plane = Plane(lens_pos, source_plane.normal)
    rays += abbe_lens(source, lens_plane, f)

    # End plane
    end_normal = rays[-1].direction[0, 0, :]
    end_pos = lens_pos + end_plane_to_lens*z
    end_plane = Plane(end_pos, end_normal)
    rays += [rays[-1].intersect_plane(end_plane)]

    assert comparetensors(rays[-1].pathlength_m.std(), 0)

    # Backpropagate
    rays_back = abbe_lens(rays[-1], lens_plane, f)
    rays += rays_back
    rays += [rays[-1].intersect_plane(source_plane)]

    # # Uncomment for debugging
    # # ### Plot ###
    # from matplotlib import pyplot as plt
    # from plot_functions import plot_plane, plot_lens, plot_rays
    # plt.figure()
    # ax = plt.gca()
    # plot_rays(ax, rays)
    # plot_plane(ax, source_plane, text1='src')
    # plot_lens(ax, lens_plane, f, scale=0.3)
    # plot_plane(ax, end_plane, scale=0.2, text1='end plane')
    # ax.set_aspect(1)
    # plt.show()
    # # ############

    assert comparetensors(rays[-1].position_m.std(dim=(-2, -3)), 0)
    assert comparetensors(rays[-1].pathlength_m, 0)


def test_pathlength4():
    """
    Test pathlength for collimated source and abbe sine lens in media with refractive index > 1.
    """
    origin, x, y, z = cartesian3d()

    # Media
    n1 = 1.6
    n2 = 1.2

    # Source
    Nx = 1
    Ny = 7
    beam_width = 0.15
    source_pos = origin + 0.05*x
    y_rot = rotate(y, x, -0.15)
    source_plane = CoordPlane(source_pos, beam_width*x, beam_width*y_rot)
    source = collimated_source(source_plane, Nx, Ny, refractive_index=n1)

    # Lens
    f = 0.3
    lens_pos = origin + f*z
    lens_plane = Plane(lens_pos, z)
    rays_at_lens = abbe_lens(source, lens_plane, f, n_out=n2)

    # Back focal plane
    end_pos = lens_pos + f*z
    BFP = Plane(end_pos, z)
    end_ray = rays_at_lens[1].intersect_plane(BFP)

    # Backpropagate
    rays_back_at_lens = abbe_lens(end_ray, lens_plane, f, n_out=n1)
    back_to_the_start = rays_back_at_lens[1].intersect_plane(source_plane)

    # # Uncomment plot code for debugging
    # ### Plot ###
    # from matplotlib import pyplot as plt
    # from plot_functions import plot_plane, plot_lens, plot_rays
    # fig = plt.figure()
    # ax = plt.gca()
    # plt.title('test_pathlength4')
    # plot_plane(ax, source_plane, scale=1.2, text1=f' src\n n1={n1}')
    # plot_lens(ax, lens_plane, f, scale=0.3)
    # plot_plane(ax, BFP, scale=0.3, text1=f' BFP\n n2={n2}')
    # plot_rays(ax, [source, *rays_at_lens, end_ray, *rays_back_at_lens, back_to_the_start])
    # plot_rays(ax, rays_at_lens, plotkwargs={'color': 'red', 'marker': 'o', 'linestyle': ''})
    # ax.set_aspect(1)
    # plt.xlabel('z (m)')
    # plt.ylabel('y (m)')
    # plt.show()
    # ############

    assert comparetensors(end_ray.pathlength_m.std(), 0)
    assert comparetensors(end_ray.position_m.std(dim=(-2, -3)), 0)
    assert comparetensors(back_to_the_start.pathlength_m, 0)
    assert comparetensors(back_to_the_start.direction.std(dim=(-2, -3)), 0)


def test_pathlength5():
    """
    Test pathlength for point source and abbe sine lens in media with refractive index > 1.
    """
    origin, x, y, z = cartesian3d()

    # Media
    n1 = 1.3
    n2 = 1.0

    # Source
    Nx = 1
    Ny = 7
    beam_opening = 1.2
    source_pos = origin
    source_plane = CoordPlane(source_pos, beam_opening*x, beam_opening*y)
    source = point_source(source_plane, Nx, Ny, refractive_index=n1)

    # Lens
    f = 1.7e-3
    lens_pos = origin + f*z
    lens_plane = Plane(lens_pos, z)
    rays_at_lens = abbe_lens(source, lens_plane, f, n_out=n2)

    # Back focal plane
    end_pos = lens_pos + f*z
    BFP = Plane(end_pos, z)
    end_ray = rays_at_lens[1].intersect_plane(BFP)

    # Backpropagate
    rays_back_at_lens = abbe_lens(end_ray, lens_plane, f, n_out=n1)
    back_to_the_start = rays_back_at_lens[1].intersect_plane(source_plane)

    # # Uncomment plot code for debugging
    # ### Plot ###
    # from matplotlib import pyplot as plt
    # from plot_functions import plot_plane, plot_lens, plot_rays
    # fig = plt.figure()
    # ax = plt.gca()
    # plt.title('test_pathlength5')
    # plot_plane(ax, source_plane, scale=f, text1=f' src\n n1={n1}')
    # plot_lens(ax, lens_plane, f, scale=f)
    # plot_plane(ax, BFP, scale=f, text1=f' BFP\n n2={n2}')
    # plot_rays(ax, [source, *rays_at_lens, end_ray, *rays_back_at_lens, back_to_the_start])
    # plot_rays(ax, rays_at_lens, plotkwargs={'color': 'red', 'marker': 'o', 'linestyle': ''})
    # ax.set_aspect(1)
    # plt.xlabel('z (m)')
    # plt.ylabel('y (m)')
    # plt.show()
    # ############

    assert comparetensors(end_ray.pathlength_m, f*(n1+n2))
    assert comparetensors(end_ray.direction.std(dim=(-2, -3)), 0)
    assert comparetensors(back_to_the_start.pathlength_m, 0)
    assert comparetensors(back_to_the_start.position_m.std(dim=(-2, -3)), 0)
