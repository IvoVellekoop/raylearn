"""Field-related functions

Functions for converting sets of rays to a field with amplitude and phase
"""

import numpy as np
import torch

from ray_plane import Ray, CoordPlane
from interpolate import interpolate2d


def pathlength_to_phase(pathlength_m, wavelength_m, amplitude=1.):
    """
    Calculate the phase of the light from the optical path length

    """
    k = 2*np.pi / wavelength_m
    field = amplitude * np.exp(1j * k * pathlength_m)
    return field

def field_from_rays(rays, camera_plane, field_coords, wavelength_m=1e-6):
    """
    Take a set of rays and calculate the field at a given plane

    WIP, does not account for ray intensity/field amplitude yet
    """
    cam_coords = camera_plane.transform(rays)
    values_out_unfolded, mask = interpolate2d(cam_coords, rays.pathlength_m, field_coords)

    field = pathlength_to_phase(values_out_unfolded, wavelength_m)
    field_out = torch.sum(mask * field, (0, 1, 2))
    return field_out

def coord_grid(limits=(-1., 1., -1., 1.), resolution=(100,100)):
    """
    Create an X-by-Y-by-2 tensor with coordinates to be used for interpolation
    """
    x_array = torch.tensor((1,0)) * torch.linspace(limits[0], limits[1], resolution[0]).view(1, resolution[0], 1)
    y_array = torch.tensor((0,1)) * torch.linspace(limits[2], limits[3], resolution[1]).view(resolution[1], 1, 1)

    field_coords = x_array + y_array

    return field_coords