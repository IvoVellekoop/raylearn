"""Optical functions.

Functions that represent optical elements. They have Ray objects as input and/or output.

Terminology used:
('Mx...' denotes one or multiple dimensions of undefined length.)
Vector: A DxMx... torch tensor where the 0th dimension denotes vector components.
        Usually, D=3, representing vector dimensions x, y and z. Other dimensions are used for
        parallelization.
Scalar: A 1xMx... torch tensor where the first dimension has length 1. Or, in cases where all Ray
        elements have the same value, this may be replaced by a Python float or double. Other
        dimensions are used for parallellization.
Note that for vector and scalar operations, the number of dimensions must match! Dimensionality of
e.g. 3x1x1 is not the same as 3 for torch tensors, and will yield incorrect results for the vector
operations!
"""

import torch
import numpy as np
from vector_functions import dot, unit, norm_square, rejection
from ray_plane import Ray, Plane


def collimated_source(sourceplane, Nx, Ny, **raykwargs):
    """###WIP! Define size differently for more dimensions. Collimated source.
    Return a Ray object with 3 by Nx by Ny position_m. Position and grid spacing determined by
    input sourceplane and Nx & Ny.

    Input
    -----
        sourceplane     CoordPlane. Position_m defines the center position. The x and y vectors
                        define the span of the rays: distance in meter from center to outer ray.
        Nx              Number of ray elements along x plane direction.
        Ny              Number of ray elements along y plane direction.
        raykwargs       Additional properties to pass to the Ray object.
    """
    x_array = sourceplane.x * torch.linspace(-1.0, 1.0, Nx).view(Nx, 1, 1)
    y_array = sourceplane.y * torch.linspace(-1.0, 1.0, Ny).view(1, Ny, 1)
    pos = sourceplane.position_m + x_array + y_array

    return Ray(pos, sourceplane.normal, **raykwargs)


def point_source(sourceplane, Nx, Ny, **raykwargs):
    """Point source with limited opening angle.
    Return a Ray object with 3 by Nx by Ny position_m. Position and grid spacing determined by
    input sourceplane and Nx & Ny.

    Input
    -----
        sourceplane     CoordPlane. Position_m defines the center position. The x and y vectors
                        define the aperture of the rays: tan(angle) between center and outer ray.
        Nx              Number of ray elements along x plane direction.
        Ny              Number of ray elements along y plane direction.
        raykwargs       Additional properties to pass to the Ray object.
    """
    x_array = sourceplane.x * torch.linspace(-1.0, 1.0, Nx).view(Nx, 1, 1)
    y_array = sourceplane.y * torch.linspace(-1.0, 1.0, Ny).view(1, Ny, 1)
    direction = unit(sourceplane.normal + x_array + y_array)

    return Ray(sourceplane.position_m, direction, **raykwargs)


def ideal_lens(in_ray, lens, f):
    """WIP! pathlength not implemented. Ideal infinitely thin lens.

    Input
    -----
        in_ray:  Ray object. Input Ray.
        lens:    Plane object. Lens plane.
        f:       Scalar array. Focal distance.

    Output
    ------
        out_ray: Ray object. Output Ray.

    """
    OC       = lens.position_m                            # Optical Center position
    BFP      = Plane(OC - f * lens.normal, lens.normal)   # Back Focal Plane
    P        = in_ray.intersect_plane(lens).position_m    # Ray intersection with lens plane
    chiefray = Ray(OC, in_ray.direction)                  # Chief or Principal Ray (through OC)
    focus    = chiefray.intersect_plane(BFP).position_m   # Ray intersection with Back Focal Plane

    ### Compute pathlength
    # A = in_ray_plane_intersect(in_ray_pos, in_ray_dir, FFP, lens_dir)   # Ray intersection with front focal plane
    # K = in_ray_plane_intersect(P, new_in_ray_dir, BFP, new_in_ray_dir)  # Ray intersection with isophase plane at BFP
    # Delta = 2*f - norm(A-P) - norm(P-K)
    #     Delta = f - torch.sqrt(f*f + normsq(P-L))

    out_ray = Ray(P, unit(focus - P))                     # Output Ray
    return out_ray


def smooth_grid(x, y, spacing_x, spacing_y, power):
    """Smooth Grid function.

    Input
    -----
    """
    cosinegrid, indices = torch.max(torch.stack((
        (0.5 * torch.cos(x * 2 * np.pi / spacing_x) + 0.5),
        (0.5 * torch.cos(y * 2 * np.pi / spacing_y) + 0.5))), dim=-1)

    return 1 - cosinegrid ** power


def intensity_mask_smooth_grid(in_ray, coordplane, spacing_x, spacing_y, power):
    """Intensity Mask of Smooth Grid.

    Pass rays through intensity mask.
    """
    rays_at_plane = in_ray.intersect_plane(coordplane)
    x, y = coordplane.transform(rays_at_plane)
    factors = smooth_grid(x, y, spacing_x, spacing_y, power)
    return rays_at_plane.mask(factors)


def snells(ray_in, N, n_out):
    """
    Compute refracted Ray.

    Returns a new Ray instance with the new direction and refractive index.

    Input
    -----
        ray_in      Ray. Incoming Ray.
        N           Unit vector. Surface normal of interface.
        n_out       Refractive index outgoing ray

    Output
    ------
        ray_out     Ray. Refracted outgoing Ray.

    Notes on derivation: rejection(ray_dir, surface_normal) has magnitude sin(angle)
    and thus it can act as sine in Snell's law. Furthermore, it is perpendicular to
    the surface_normal. Hence, it's the perpendicular component of dir_out. Once
    this is known, the parallel component can also be computed, since |dir_out| = 1.
    """
    dir_in = ray_in.direction
    n_in = ray_in.refractive_index

    dir_inrej = rejection(dir_in, N)       # Perpendicular component dir_in
    dir_outrej = n_in/n_out * dir_inrej    # Perpendicular component dir_out
    dir_out = dir_outrej - N * torch.sqrt(1 - norm_square(dir_outrej))

    ray_out = ray_in.copy(direction=dir_out, refractive_index=n_out)
    return ray_out
