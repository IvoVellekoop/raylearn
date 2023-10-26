"""
Materials.

Define material properties as functions, classes, or values.
"""

from torch import tensor
from refractive_index import refractive_index_csv, refractive_index_sellmeier, refractive_index_cauchy
from vector_functions import ensure_tensor
from dirconfig_raylearn import dirs


def n_water(wavelength_m):
    """
    Compute refractive index of distilled water (20°C) for specified wavelengths (in m).
    Sellmeier coefficients from Table 2 of [Daimon and Masumura 2007, DOI:10.1364/AO.46.003811].
    The input wavelength_m can be a value, or array of any size, and is automatically converted to
    tensor.
    """
    wavelength_um = ensure_tensor(wavelength_m * 1e6)
    assert((wavelength_um > 0.1817).all())      # Check valid wavelength range
    assert((wavelength_um < 1.129).all())       # Check valid wavelength range

    # Sellmeier coefficients
    B = tensor((5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1))
    C = tensor((5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1))     # in µm²

    return refractive_index_sellmeier(wavelength_um, B, C)


def n_PBS(wavelength_m):
    """
    Compute refractive index of Phosphate Buffered Saline (PBS) at room temperature for specified
    wavelengths (in m).
    Sellmeier coefficients from Table 2 of [Hoang et al. 2019, DOI:10.3390/app9061145].
    """

    wavelength_um = ensure_tensor(wavelength_m * 1e6)
    assert((wavelength_um > 0.450).all())       # Check valid wavelength range
    assert((wavelength_um < 1.400).all())       # Check valid wavelength range

    # Sellmeier coefficients
    B = tensor((0.763614, -9246.06, 47.7781))
    C = tensor((0.00905, 30036.29, 151.858))    # in µm²

    return refractive_index_sellmeier(wavelength_um, B, C)


def n_collagen(wavelength_m):
    """
    Compute refractive index of collagen at room temperature for specified wavelengths (in m).
    Cauchy coefficients from Equation 2 of [Bashkatov et al. 2000, DOI:10.1117/12.405952].
    """

    wavelength_um = ensure_tensor(wavelength_m * 1e6)

    # Cauchy coefficients
    A = 1.426
    B = 19476e-6            # in µm²
    C = -1131066900e-12     # in µm⁴
    return refractive_index_cauchy(wavelength_um, A, B, C)


def n_SodaLimeGlass(wavelength_m):
    """
    Get refractive index of soda lime glass at specified wavelengths (in m).
    """

    wavelength_um = ensure_tensor(wavelength_m * 1e6)
    assert((wavelength_um > 0.249).all())       # Check valid wavelength range
    assert((wavelength_um < 1.8).all())         # Check valid wavelength range
    csvpath = dirs['repo'].joinpath('data/n-sodalime-vogt2016.csv')
    return refractive_index_csv(wavelength_um, csvpath)
