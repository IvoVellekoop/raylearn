"""
Learn the thickness of a piece of glass in the Two Photon Microscope.
"""

import torch
from torch import tensor
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torchviz import make_dot

from plot_functions import plot_coords, format_prefix
from testing import MSE
from tpm import TPM


# Set default tensor type to double (64 bit)
# Machine epsilon of float (32-bit) is 2^-23 = 1.19e-7
# The ray simulation contains both meter and micrometer scales,
# hence floats might not be precise enough.
# https://en.wikipedia.org/wiki/Machine_epsilon
torch.set_default_tensor_type('torch.DoubleTensor')

plt.rc('font', size=12)

# Import measurement
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-400um/raylearn_pencil_beam_738477.786123_400um.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-empty/raylearn_pencil_beam_738477.729080_empty.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/10-Feb-2022-empty/raylearn_pencil_beam_738562.645439_empty.mat'
matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/01-Jun-2022-170um_aligned_to_galvos/raylearn_pencil_beam_738673.606682_170um_aligned_to_galvos.mat' ###### fix path
matfile = h5py.File(matpath, 'r')

cam_ft_coords_gt = tensor((matfile['cam_ft_col'], matfile['cam_ft_row'])).permute(1, 2, 0)
cam_im_coords_gt = tensor((matfile['cam_img_col'], matfile['cam_img_row'])).permute(1, 2, 0)

# Create TPM object and perform initial raytrace
tpm = TPM()
tpm.set_measurement(matfile)
tpm.update()
tpm.raytrace()

# Define Inital Guess
tpm.slm_zshift = tensor((0.,), requires_grad=True)
tpm.slm_angle = tensor((0.,), requires_grad=True)
tpm.galvo_angle = tensor((0.,), requires_grad=True)
tpm.cam_ft_xshift = tensor((0.,), requires_grad=True)
tpm.cam_ft_yshift = tensor((0.,), requires_grad=True)
tpm.cam_im_xshift = tensor((0.,), requires_grad=True)
tpm.cam_im_yshift = tensor((0.,), requires_grad=True)
tpm.cam_im_zshift = tensor((0.,), requires_grad=True)
tpm.sample_zshift = tensor((0.,), requires_grad=True)
tpm.obj2_zshift = tensor((0.,), requires_grad=True)
tpm.L9_zshift = tensor((0.,), requires_grad=True)
tpm.L10_zshift = tensor((0.,), requires_grad=True)
tpm.coverslip_thickness = tensor((170e-6,), requires_grad=True)

# Parameter groups
params = {}
params['angle'] = {
    'SLM angle': tpm.slm_angle,
    'Galvo angle': tpm.galvo_angle,
}
params['objective'] = {
    # 'OBJ2 zshift': tpm.obj2_zshift,
    # 'sample zshift': tpm.sample_zshift,
}
params['other'] = {
    'SLM zshift': tpm.slm_zshift,
    # 'L9 zshift': tpm.L9_zshift,
    'cam ft xshift': tpm.cam_ft_xshift,
    'cam ft yshift': tpm.cam_ft_yshift,
    'cam im xshift': tpm.cam_im_xshift,
    'cam im yshift': tpm.cam_im_yshift,
    # 'cam im zshift': tpm.cam_im_zshift,
}

tpm.update()

# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 1.0e-3, 'params': params['angle'].values()},
        {'lr': 2.0e-4, 'params': params['objective'].values()},
        {'lr': 1.0e-3, 'params': params['other'].values()},
    ], lr=1.0e-5)

iterations = 200
errors = torch.zeros(iterations)


# Initialize logs for tracking each parameter
params_logs = {}
for groupname in params:
    params_logs[groupname] = {}
    for paramname in params[groupname]:
        params_logs[groupname][paramname] = torch.zeros(iterations)


trange = tqdm(range(iterations), desc='error: -')

# Plot
fig, ax = plt.subplots(nrows=2, figsize=(5, 10), dpi=110)

fig_tpm = plt.figure(figsize=(15, 4), dpi=110)
ax_tpm = plt.gca()


for t in trange:
    # === Learn === #
    # Forward pass
    tpm.update()
    cam_ft_coords, cam_im_coords = tpm.raytrace()

    # Compute and print error
    error = MSE(cam_ft_coords_gt, cam_ft_coords) \
        + MSE(cam_im_coords_gt, cam_im_coords) \
        + torch.std(cam_ft_coords, 1).sum()             # Minimize spread from Galvo tilts

    error_value = error.detach().item()
    errors[t] = error_value

    for groupname in params:
        for paramname in params[groupname]:
            params_logs[groupname][paramname][t] = params[groupname][paramname].detach().item()

    trange.desc = f'error: {error_value:<8.3g}' \
        + f'slm zshift: {format_prefix(tpm.slm_zshift, "8.3f")}m'

    # error.backward(retain_graph=True)
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    if t % 50 == 0 and True:
        plt.figure(fig.number)

        # Fourier cam
        cam_ft_coord_pairs_x, cam_ft_coord_pairs_y = \
                torch.stack((cam_ft_coords_gt, cam_ft_coords)).detach().unbind(-1)

        ax[0].clear()
        ax[0].plot(cam_ft_coord_pairs_x.view(2, -1), cam_ft_coord_pairs_y.view(2, -1),
                   color='lightgrey')
        plot_coords(ax[0], cam_ft_coords_gt[:, :, :], {'label': 'measured'})
        plot_coords(ax[0], cam_ft_coords[:, :, :], {'label': 'sim'})

        ax[0].set_ylabel('y (pix)')
        ax[0].set_xlabel('x (pix)')
        ax[0].legend(loc=1)
        ax[0].set_title(f'Fourier Cam | slm zshift={format_prefix(tpm.slm_zshift)}m | iter: {t}')

        # Image cam
        cam_im_coord_pairs_x, cam_im_coord_pairs_y = \
            torch.stack((cam_im_coords_gt, cam_im_coords)).detach().unbind(-1)

        ax[1].clear()
        ax[1].plot(cam_im_coord_pairs_x.view(2, -1), cam_im_coord_pairs_y.view(2, -1),
                   color='lightgrey')
        plot_coords(ax[1], cam_im_coords_gt[:, :, :], {'label': 'measured'})
        plot_coords(ax[1], cam_im_coords[:, :, :], {'label': 'sim'})

        ax[1].set_ylabel('y (pix)')
        ax[1].set_xlabel('x (pix)')
        ax[1].legend(loc=1)
        ax[1].set_title(f'Image Cam | iter: {t}')

        plt.draw()
        plt.pause(1e-3)

        plt.figure(fig_tpm.number)
        ax_tpm.clear()
        tpm.plot(ax_tpm)
        plt.draw()
        plt.pause(1e-3)


for groupname in params:
    print('\n' + groupname + ':')
    for paramname in params[groupname]:
        if groupname == 'angle':
            print(f'  {paramname}: {params[groupname][paramname].detach().item():.3f}rad')
        else:
            print(f'  {paramname}: {format_prefix(params[groupname][paramname])}m')


fig, ax1 = plt.subplots(figsize=(7, 7))
fig.dpi = 144

# Plot error
errorcolor = 'darkred'
RMSEs = np.sqrt(errors.detach().cpu())
ax1.plot(RMSEs, label='error', color=errorcolor)
ax1.set_ylabel('Error (pix)')
ax1.set_ylim((0, max(RMSEs)))
ax1.legend(loc=2)

ax2 = ax1.twinx()
for groupname in params:
    for paramname in params_logs[groupname]:
        ax2.plot(params_logs[groupname][paramname], label=paramname)
ax2.set_ylabel('Parameter (m | rad)')
ax2.legend(loc=1)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Learning parameters')
plt.show()

# === Glass plate === #

tpm.set_measurement(matfile)
tpm.update()
tpm.raytrace()


# Import measurement
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-400um/raylearn_pencil_beam_738477.768870_400um.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-empty/raylearn_pencil_beam_738477.729080_empty.mat'
# matpath = "F:/ScientificData/pencil-beam-positions/01-Jun-2022-170um+400um_aligned_to_galvos/raylearn_pencil_beam_738673.636565_170um+400um_aligned_to_galvos.mat"
matpath = "F:/ScientificData/pencil-beam-positions/01-Jun-2022-400um_aligned_to_slm/raylearn_pencil_beam_738673.696695_400um_aligned_to_slm.mat"
matfile = h5py.File(matpath, 'r')

cam_ft_coords_gt = tensor((matfile['cam_ft_col'], matfile['cam_ft_row'])).permute(1, 2, 0)
cam_im_coords_gt = tensor((matfile['cam_img_col'], matfile['cam_img_row'])).permute(1, 2, 0)

# Parameters
tpm.coverslip_thickness = tensor((600e-6,), requires_grad=True)

params_coverslip = {
    'Coverslip Thickness': tpm.coverslip_thickness,
    'cam im xshift': tpm.cam_im_xshift,
    'cam im yshift': tpm.cam_im_yshift,
}

tpm.set_measurement(matfile)
tpm.update()

# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 1.0e-4, 'params': params_coverslip.values()},
    ], lr=1.0e-5)

iterations = 200
errors = torch.zeros(iterations)


params_coverslip_log = {}
for name in params_coverslip:
    params_coverslip_log[name] = torch.zeros(iterations)

trange = tqdm(range(iterations), desc='error: -')


# Plot
fig, ax = plt.subplots(nrows=2, figsize=(5, 10))
fig.dpi = 144

for t in trange:
    # === Learn === #
    # Forward pass
    tpm.update()
    cam_ft_coords, cam_im_coords = tpm.raytrace()

    # Compute and print error
    error = MSE(cam_ft_coords_gt, cam_ft_coords) \
        + MSE(cam_im_coords_gt, cam_im_coords) \

    error_value = error.detach().item()
    errors[t] = error_value

    for name in params_coverslip:
        params_coverslip_log[name][t] = params_coverslip[name][-1].detach().item()

    trange.desc = f'error: {error_value:<8.3g}' \
        + f'coverslip thickness: {format_prefix(tpm.coverslip_thickness, "8.3f")}m'

    # error.backward(retain_graph=True)
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Plot
    if t % 10 == 0 and True:
        # Fourier cam
        cam_ft_coord_pairs_x, cam_ft_coord_pairs_y = \
                torch.stack((cam_ft_coords_gt, cam_ft_coords)).detach().unbind(-1)

        ax[0].clear()
        ax[0].plot(cam_ft_coord_pairs_x.view(2, -1), cam_ft_coord_pairs_y.view(2, -1),
                   color='lightgrey')
        plot_coords(ax[0], cam_ft_coords_gt[:, :, :], {'label': 'measured'})
        plot_coords(ax[0], cam_ft_coords[:, :, :], {'label': 'sim'})

        ax[0].set_ylabel('y (pix)')
        ax[0].set_xlabel('x (pix)')
        ax[0].legend(loc=1)
        ax[0].set_title(f'Fourier Cam | coverslip={format_prefix(tpm.coverslip_thickness)}m | iter: {t}')

        # Image cam
        cam_im_coord_pairs_x, cam_im_coord_pairs_y = \
            torch.stack((cam_im_coords_gt, cam_im_coords)).detach().unbind(-1)

        ax[1].clear()
        ax[1].plot(cam_im_coord_pairs_x.view(2, -1), cam_im_coord_pairs_y.view(2, -1),
                   color='lightgrey')
        plot_coords(ax[1], cam_im_coords_gt[:, :, :], {'label': 'measured'})
        plot_coords(ax[1], cam_im_coords[:, :, :], {'label': 'sim'})

        ax[1].set_ylabel('y (pix)')
        ax[1].set_xlabel('x (pix)')
        ax[1].legend(loc=1)
        ax[1].set_title(f'Image Cam | iter: {t}')

        plt.draw()
        plt.pause(1e-3)


print(f'\ncoverslip thickness: {format_prefix(tpm.coverslip_thickness)}m\n')

plt.show()

tpm.plot()


fig, ax1 = plt.subplots(figsize=(7, 7))
fig.dpi = 144


# Plot error
errorcolor = 'tab:red'
RMSEs = np.sqrt(errors.detach().cpu())
ax1.plot(RMSEs, label='error', color=errorcolor)
ax1.set_ylabel('Error (pix)')
ax1.set_ylim((0, max(RMSEs)))
ax1.legend()

ax2 = ax1.twinx()
for name in params_coverslip_log:
    ax2.plot(params_coverslip_log[name]*1e6, label=name)
ax2.set_ylabel('Parameter (um | rad)')
ax2.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Learning parameters')
plt.show()


# ##### === Check radial error relation ===
# error_per_pencil = ((cam_ft_coords - cam_ft_coords_gt).mean(dim=0) ** 2).sum(dim=1).sqrt().view(-1).detach()
# dist_to_center = (cam_ft_coords_gt.mean(dim=0) ** 2).sum(dim=1).sqrt().view(-1).detach()
#
# fig = plt.figure(figsize=(15, 4))
# fig.dpi = 144
# ax1 = plt.gca()
#
# plt.plot(dist_to_center, error_per_pencil, '.')
# plt.xlabel('Distance to center')
# plt.ylabel('Error per pencil beam')
#
# plt.show()
# ##### ===================================
# pass