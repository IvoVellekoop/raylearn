import torch
from torch import tensor
import numpy as np
import h5py
import hdf5storage
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torchviz import make_dot

from plot_functions import plot_coords, format_prefix, plot_rays, plot_lens, plot_plane, default_viewplane
from testing import MSE
from tpm import TPM
from vector_functions import components, rejection
from interpolate_shader import interpolate_shader
from optical import Coverslip
from sample_tube import SampleTube
from ray_plane import CoordPlane
from dirconfig_raylearn import dirs

# Create synthetic version

# Define 'measurement' Galvo and SLM settings
matfile = {
    'p/rects': 0.3 * torch.tensor(((0., 1, 0, 0), (-1, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, -1, 0, 0))).T,
    'p/galvoXs': ((0, -0.04, 0, 0.04, 0),),
    'p/galvoYs': ((0.04, 0, 0, 0, -0.04),),
    'p/GalvoXcenter': (0.,),
    'p/GalvoYcenter': (0.,)}

tpm = TPM()

tpm.set_measurement(matfile)
tpm.sample = SampleTube()
tpm.sample_zshift = tensor((0.,))
tpm.sample.tube_angle = tensor((np.radians(90.),))

tpm.update()
cam_ft_coords_synth_gt, cam_im_coords_synth_gt = tpm.raytrace()

tpm.sample.tube_angle = tensor((np.radians(90.),))

# Size of parameter scan
N_sz = 25
N_oz = 25
N_ir = 25
N_or = 25
n_sz = int(np.ceil(N_sz/2))-1
n_oz = int(np.ceil(N_oz/2))-1
n_ir = int(np.ceil(N_ir/2))-1
n_or = int(np.ceil(N_or/2))-1

# Create parameter scan
# tpm.sample_zshift = torch.linspace(-40e-6, 40e-6, N_sz).view(*((-1,) + (1,)*3))
# tpm.obj2_zshift = torch.linspace(-40e-6, 40e-6, N_oz).view(*((-1,) + (1,)*4))
# tpm.sample.inner_radius_m = torch.linspace(280e-6, 395e-6, N_ir).view(*((-1,) + (1,)*5))
# tpm.sample.outer_radius_m = torch.linspace(400e-6, 550e-6, N_or).view(*((-1,) + (1,)*6))

tpm.sample_zshift = torch.linspace(-10e-6, 10e-6, N_sz).view(*((-1,) + (1,)*3))
tpm.obj2_zshift = torch.linspace(-10e-6, 10e-6, N_oz).view(*((-1,) + (1,)*4))
tpm.sample.inner_radius_m = torch.linspace(320e-6, 380e-6, N_ir).view(*((-1,) + (1,)*5))
tpm.sample.outer_radius_m = torch.linspace(450e-6, 520e-6, N_or).view(*((-1,) + (1,)*6))

# Closest to ground truth
fresh_tube = SampleTube()
i_inner_radius_m = (tpm.sample.inner_radius_m - fresh_tube.inner_radius_m).abs().argmin()
i_outer_radius_m = (tpm.sample.outer_radius_m - fresh_tube.outer_radius_m).abs().argmin()

tpm.update()
cam_ft_coords_synth, cam_im_coords_synth = tpm.raytrace()

# Compute and print error
error = MSE(cam_ft_coords_synth_gt, cam_ft_coords_synth, dim=(-1, -2, -3)) \
    + MSE(cam_im_coords_synth_gt, cam_im_coords_synth, dim=(-1, -2, -3)) \
    # + torch.std(cam_ft_coords_synth, 1).sum()             # Minimize spread from Galvo tilts

# fig_tpm = plt.figure(figsize=(6, 6), dpi=110)
# plt.imshow(error[i_outer_radius_m, i_inner_radius_m, :, :].squeeze(),
#     extent=(tpm.sample_zshift.min()*1e3, tpm.sample_zshift.max()*1e3, tpm.obj2_zshift.min()*1e3, tpm.obj2_zshift.max()*1e3))
# plt.xlabel('sample z-shift (mm)')
# plt.ylabel('OBJ2 z-shift (mm)')
# plt.title('Error')
# plt.colorbar()
# plt.show()


fig_tpm = plt.figure(figsize=(6, 5), dpi=110)
errorslice = error[:, :, n_oz, n_sz].squeeze()
plt.imshow(errorslice, origin='lower',
    extent=(tpm.sample.inner_radius_m.min()*1e3, tpm.sample.inner_radius_m.max()*1e3, tpm.sample.outer_radius_m.min()*1e3, tpm.sample.outer_radius_m.max()*1e3))
plt.colorbar()

plt.contour(tpm.sample.inner_radius_m.squeeze()*1e3, tpm.sample.outer_radius_m.squeeze()*1e3, errorslice, levels=25, colors='k')
plt.plot(fresh_tube.inner_radius_m*1e3, fresh_tube.outer_radius_m*1e3, '+w')
plt.text(fresh_tube.inner_radius_m*1e3, fresh_tube.outer_radius_m*1e3, ' Ground\n truth', color='lightgrey')

plt.xlabel('inner radius (mm)')
plt.ylabel('outer radius (mm)')
plt.title('Error on synthetic data')
plt.show()



fig_tpm = plt.figure(figsize=(6, 5), dpi=110)
errorslice = error[:, i_inner_radius_m, n_oz, :].squeeze()
plt.imshow(errorslice, origin='lower',
    extent=(tpm.sample_zshift.min()*1e6, tpm.sample_zshift.max()*1e6, tpm.sample.outer_radius_m.min()*1e3, tpm.sample.outer_radius_m.max()*1e3), aspect='auto')
plt.colorbar()

plt.contour(tpm.sample_zshift.squeeze()*1e6, tpm.sample.outer_radius_m.squeeze()*1e3, errorslice, levels=25, colors='k', aspect='auto')
plt.plot(0, fresh_tube.outer_radius_m*1e3, '+w')
plt.text(0, fresh_tube.outer_radius_m*1e3, ' Ground\n truth', color='lightgrey')

plt.xlabel('Sample z-shift (um)')
plt.ylabel('outer radius (mm)')
plt.title('Error on synthetic data')
plt.show()


fig_tpm = plt.figure(figsize=(6, 5), dpi=110)
errorslice = error.min(dim=1).values.min(dim=1).values
plt.imshow(errorslice, origin='lower',
    extent=(tpm.sample_zshift.min()*1e6, tpm.sample_zshift.max()*1e6, tpm.sample.outer_radius_m.min()*1e3, tpm.sample.outer_radius_m.max()*1e3), aspect='auto')
plt.colorbar()

# plt.contour(tpm.sample_zshift.squeeze()*1e6, tpm.sample.outer_radius_m.squeeze()*1e3, errorslice, levels=25, colors='k', aspect='auto')
plt.plot(0, fresh_tube.outer_radius_m*1e3, '+w')
plt.text(0, fresh_tube.outer_radius_m*1e3, ' Ground\n truth', color='lightgrey')

plt.xlabel('Sample z-shift (um)')
plt.ylabel('outer radius (mm)')
plt.title('Error on synthetic data\nfor inner radius and obj zshift: pick values to minimize error')
plt.show()

pass
