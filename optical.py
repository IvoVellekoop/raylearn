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
from torch import nn
from vector_functions import unit, vector
from ray_plane import Ray, Plane


def point_source():
    """WIP! Point source."""
    Nx = 5
    Ny = 1
    mult_opening = 10e-3;
    p0x, p0y = torch.meshgrid(mult_opening*torch.linspace(-1.0,1.0,Nx), \
                              mult_opening*torch.linspace(-1.0,1.0,Ny))
    p0z = torch.zeros(Ny,Nx)
    p0 = torch.stack((p0x.view(Ny,Nx), p0y.view(Ny,Nx), p0z.view(Ny,Nx)))

    p1 = vector((0,0,0), (3, Ny, Nx))
    s0 = unit(p1 - p0)

    return Ray(p1, s0)

point_source()


def collimated_source(size):
    """WIP! Collimated source."""
    mult_opening = 10e-3;

    posx, posy = torch.meshgrid( mult_opening * torch.linspace(-1.0,1.0,size[1]), \
                                 mult_opening * torch.linspace(-1.0,1.0,size[2]))
    posz = torch.zeros(size[1:])
    pos = torch.stack((posx, posy, posz))

    d = vector((0,0,1), size)

    # Turn into parameter
    pos_param = nn.Parameter(pos, requires_grad=True)
    d_param = nn.Parameter(d, requires_grad=True)

    return Ray(pos_param, d_param)


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
    BFP      = Plane(OC - f*lens.normal, lens.normal)     # Back Focal Plane
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
