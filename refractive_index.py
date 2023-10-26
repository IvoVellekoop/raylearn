"""
Functions to compute refractive index.
Note: since most sources express their wavelengths in micrometers, these functions do too.
"""

import numpy as np
import pandas as pd
from torch import tensor


def refractive_index_sellmeier(wavelength_um, B, C):
    """
    Sellmeier equation for computing refractive index.
    See also: https://en.wikipedia.org/wiki/Sellmeier_equation

    Input
    -----
        wavelength_um   Tensor of any shape. Wavelength in micrometers (µm).
        B               Tensor of shape (N) containing Sellmeier B coefficients.
        C               Tensor of shape (N) containing Sellmeier C coefficients in µm².

    Output
    ------
        refractive index at requested wavelength
    """

    # LaTeX equation:
    # $$ n^2(\lambda)=1+\sum_i \frac{B_i \lambda^2}{\lambda^2-C_i} $$
    # Where n is refractive index and lambda is the wavelength. In some literature, different
    # symbols are used.

    # Prepare variables
    wavelength_sq = (wavelength_um * wavelength_um).unsqueeze(-1)
    assert(B.shape == C.shape)          # Check that coefficient arrays have the same shape
    assert(B.dim() == 1)                # Check dimensionality of coefficient arrays

    # Compute refractive index
    summation = (B * wavelength_sq / (wavelength_sq - C)).sum(dim=-1)
    refractive_index = (1 + summation).sqrt()
    return refractive_index


def refractive_index_cauchy(wavelength_um, A, B, C):
    """
    Cauchy equation for refractive index.

    Input
    -----
        wavelength_um   Tensor of any shape. Wavelength in micrometers (µm)
        A, B, C         Cauchy coefficients (units: 1, µm², µm⁴)

    Output
    ------
        refractive index at requested wavelength
    """

    # Prepare variables
    wavelength_pow2 = (wavelength_um * wavelength_um).unsqueeze(-1)     # λ²
    wavelength_pow4 = wavelength_pow2 * wavelength_pow2                 # λ⁴

    # Compute refractive index
    refractive_index = A + B / wavelength_pow2 + C / wavelength_pow4
    return refractive_index


def refractive_index_csv(wavelength_um, csv_path):
    """
    Compute refractive index by linearly interpolating values from csv file.
    The header row must be: 'wavelength_um,n'. The rest of the file must contain numeric values
    only. Start and end wavelengths may be arbitrary.

    Example
    -------
        wavelength_um,n
        0.25,1.589
        0.26,1.582

    Input
    -----
        wavelength_um   Tensor of any shape. Wavelength in micrometers (µm)
        csv_path        String. File path of csv file.

    Output
    ------
        refractive index at requested wavelength
    """

    df = pd.read_csv(csv_path)
    refractive_index = np.interp(wavelength_um, df['wavelength_um'], df['n'])
    return tensor(refractive_index)
