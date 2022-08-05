"""Test interpolate_shader."""

import torch
import numpy as np

from vector_functions import cartesian3d
from ray_plane import CoordPlane
from optical import point_source
from interpolate_shader import interpolate_shader
from testing import comparetensors


torch.set_default_tensor_type('torch.DoubleTensor')


def test_phase_point_source():
    """
    Test the phase pattern of a set of rays coming from a point source.
    """
    Nx = 50
    Ny = 50
    num_pixels = 64
    screen_halfsize_m = 2e-3
    source_to_screen_distance_m = 1.0
    wavelength_m = 500e-9

    origin, x, y, z = cartesian3d()

    screen_plane = CoordPlane(origin + source_to_screen_distance_m*z, x, y)

    # Point source
    tan_angle = 1.1 * screen_halfsize_m / source_to_screen_distance_m   # Make source slightly 
    point_source_plane = CoordPlane(origin, tan_angle*x, tan_angle*y)
    point = point_source(point_source_plane, Nx, Ny, intensity=1.0)
    point_source_at_screen = point.intersect_plane(screen_plane)
    coords = screen_plane.transform_rays(point_source_at_screen)

    # Compute field with interpolate_shader
    pathlength_to_screen = point_source_at_screen.pathlength_m
    data = torch.cat((coords, pathlength_to_screen), 2)
    field_out = torch.tensor(interpolate_shader(
        data.numpy(),
        npoints=(num_pixels, num_pixels),
        limits=(-screen_halfsize_m, screen_halfsize_m, -screen_halfsize_m, screen_halfsize_m),
        wavelength_m=wavelength_m,
        ))

    # Compute analytical equivalent
    #### Note: in the current implementation, the computed field is shifted half a pixel
    #### E.g. in a 64x64, pixel (32, 32) is the central pixel.
    x_max = screen_halfsize_m
    pixelsize = 2 * screen_halfsize_m / num_pixels
    x = np.linspace(-x_max, x_max - pixelsize, num_pixels).reshape((num_pixels, 1))
    y = x.T

    # In the shader implementation the mean pathlength is subtracted to reduce the value range,
    # Therefore, there will be constant phase difference between the two computed fields
    pathlength_analytical = np.sqrt(source_to_screen_distance_m**2 + x**2 + y**2)
    field_out_analytical = np.exp(1j * pathlength_analytical * 2*np.pi / wavelength_m)

    # Compute correlation coefficient
    corrcoeff_matrix = np.corrcoef(field_out_analytical.flatten(), field_out.numpy().flatten())
    correlation = np.abs(corrcoeff_matrix[0, 1])

    assert correlation > 0.9999
