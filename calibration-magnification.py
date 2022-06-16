"""
Determine calibration matrices from the digital Two Photon Microscope.
"""

import torch
from torch import tensor
import matplotlib.pyplot as plt

from vector_functions import norm, components
from tpm import TPM


# Set default tensor type to double (64 bit)
# Machine epsilon of float (32-bit) is 2^-23 = 1.19e-7
# The ray simulation contains both meter and micrometer scales,
# hence floats might not be precise enough.
# https://en.wikipedia.org/wiki/Machine_epsilon
torch.set_default_tensor_type('torch.DoubleTensor')

plt.rc('font', size=12)


# Define 'measurement' Galvo and SLM settings
matfile = {
    'p/rects': torch.zeros((4, 1)),
    'p/galvoXs': (0.037986079230209,),
    'p/galvoYs': (0.,),
    'p/GalvoXcenter': (0.,),
    'p/GalvoYcenter': (0.,)}

# Create TPM object and perform initial raytrace
tpm = TPM()
tpm.set_measurement(matfile)
tpm.update()
tpm.raytrace()

# tpm.plot(fraction=1.)
# plt.show()

# Compute matrices
x_at_plane34_m = norm(tpm.plane34.transform_rays(tpm.plane34_ray))
x_at_plane57_m = norm(tpm.plane57.transform_rays(tpm.plane57_ray))
x_at_imgcam_m = norm(tpm.cam_im_coords * tpm.cam_pixel_size)

magnification_plane34_imgcam = x_at_imgcam_m / x_at_plane34_m
magnification_plane57_imgcam = x_at_imgcam_m / x_at_plane57_m
error_magnification_plane34_imgcam = 100 * torch.abs(magnification_plane34_imgcam - 2.841) / 2.841
error_magnification_plane57_imgcam = 100 * torch.abs(magnification_plane57_imgcam - 3.788) / 3.788

print('\nMagnifications:')
print(f'Plane34 - Image cam: {magnification_plane34_imgcam.item():.3f}  Error: {error_magnification_plane34_imgcam.item():.2f}%')
print(f'Plane57 - Image cam: {magnification_plane57_imgcam.item():.3f}  Error: {error_magnification_plane57_imgcam.item():.2f}%')


# Define 'measurement' Galvo and SLM settings
matfile2 = {
    'p/rects': torch.tensor(((0.1, 0., 0., 0.),)).T,
    'p/galvoXs': (0.,),
    'p/galvoYs': (0.,),
    'p/GalvoXcenter': (0.,),
    'p/GalvoYcenter': (0.,)}

# Create TPM object and perform initial raytrace
tpm2 = TPM()
tpm2.set_measurement(matfile2)
tpm2.update()
tpm2.raytrace()

# tpm2.plot(fraction=1.)
# plt.show()

# Compute matrices

x_at_slm_m = norm(tpm2.slm_plane.transform_rays(tpm2.slm_ray) * tpm2.slm_height)
x_at_ftcam_m = norm(tpm2.cam_ft_coords * tpm2.cam_pixel_size)

magnification_slm_ftcam = x_at_ftcam_m / x_at_slm_m
error_magnification_slm_ftcam = 100 * torch.abs(magnification_slm_ftcam - 0.3520) / 0.3520

print(f'SLM - Fourier cam: {magnification_slm_ftcam.item():.3f}  Error: {error_magnification_slm_ftcam.item():.2f}%')

pass
