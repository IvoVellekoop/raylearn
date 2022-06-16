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
    # 'p/galvoXs': (1., 0.),
    # 'p/galvoYs': (0., 1.),
    'p/galvoXs': (0.037986079230209,),
    'p/galvoYs': (0.,),
    # 'p/galvoXs': (0., 0.1,),
    # 'p/galvoYs': (0., 0.,),
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
stage_displacement_m = norm(tpm.cam_sample_coords)

zoom = 20
framesize_TPMpix = 512
galvo_angular_range_optdeg = 6 * 1.81
wavelength_m = 804e-9

slm_coords = tensor(matfile['p/rects'])[0:2, :].T.view(-1, 2)
galvo_volts = tensor((matfile['p/galvoXs'], matfile['p/galvoYs'])).T - \
    tensor((matfile['p/GalvoXcenter'], matfile['p/GalvoYcenter'])).view(1, 1, 2)

galvo_tilt_optdeg = components(galvo_volts)[0] / tpm.galvo_volts_per_optical_degree
displacement_TPMpix = galvo_tilt_optdeg * zoom * framesize_TPMpix / galvo_angular_range_optdeg


print()
print('Compare these values with tpm/calibration/tests/theoretical_galvo_slm_calibration.mlx')
print('x_stage_sample_m:', f'{stage_displacement_m.item():.4e}')
print('D_stage_TPM_pix:', displacement_TPMpix.item())

pass
