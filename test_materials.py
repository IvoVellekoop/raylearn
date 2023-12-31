"""Test materials"""

import torch
from torch import tensor
from materials import n_water, n_PBS, n_SodaLimeGlass
from testing import comparetensors


torch.set_default_tensor_type('torch.DoubleTensor')


def test_water():
    """
    Test refractive index of distilled water (20°C).  Literature values taken from Table 1 of
    [Daimon and Masumura 2007, DOI:10.1364/AO.46.003811].
    """
    precision = 1e-5         # Require 5 decimals of precision
    wavelength_m = tensor((181.78736e-9, 404.77e-9, 587.725e-9, 894.596e-9, 1083.33e-9, 1128.95e-9))
    n_literature = tensor((1.468725, 1.343113, 1.333399, 1.327068, 1.324248, 1.323559))
    n_compute = n_water(wavelength_m)
    assert(comparetensors(n_literature, n_compute, error=precision, error_in_meps=False))


def test_PBS():
    """
    Test refractive index of PBS at room temperature.  Literature value taken from Table 3 of [Hoang
    et al. 2019, DOI:10.3390/app9061145].
    """
    precision = 1e-4                # Require 4 decimals of precision
    wavelength_m = 589.3e-9         # Wavelength _D
    n_literature = 1.3348           # Refractive index n_D
    n_compute = n_PBS(wavelength_m)
    assert(comparetensors(n_literature, n_compute, error=precision, error_in_meps=False))


def test_SodaLimeGlass():
    """
    Test refractive index of Soda Lime Glass. Literature value taken from [Vogt 2016,
    10.1109/JPHOTOV.2015.2498043].
    See also: https://refractiveindex.info/?shelf=glass&book=soda-lime&page=Vogt-10ppm
    """
    precision = 1e-3                            # Require 3 decimals of precision
    wavelength_m = tensor((305e-9, 800e-9))     # Wavelength
    n_literature = tensor((1.555, 1.508))       # Refractive index
    n_compute = n_SodaLimeGlass(wavelength_m)
    assert(comparetensors(n_literature, n_compute, error=precision, error_in_meps=False))
