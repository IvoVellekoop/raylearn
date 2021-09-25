"""Field-related functions

Functions for converting sets of rays to a field with amplitude and phase
"""

import numpy as np

from optical import Ray
from interpolate import interpolate2d


def pathlength_to_phase(pathlength_m, wavelength_m):
    k = 2*np.pi / wavelength_m
    field = np.exp(1j * k * pathlength_m)
    return field
