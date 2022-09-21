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
from vector_functions import components
from interpolate_shader import interpolate_shader


# Set default tensor type to double (64 bit)
# Machine epsilon of float (32-bit) is 2^-23 = 1.19e-7
# The ray simulation contains both meter and micrometer scales,
# hence floats might not be precise enough.
# https://en.wikipedia.org/wiki/Machine_epsilon
torch.set_default_tensor_type('torch.DoubleTensor')

do_plot_empty = False
do_plot_pincushion = False
do_plot_coverslip = False
do_plot_obj1_zshift = False

plt.rc('font', size=12)

# Import measurement
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-400um/raylearn_pencil_beam_738477.786123_400um.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-empty/raylearn_pencil_beam_738477.729080_empty.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/10-Feb-2022-empty/raylearn_pencil_beam_738562.645439_empty.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/01-Jun-2022-170um_aligned_to_galvos/raylearn_pencil_beam_738673.606682_170um_aligned_to_galvos.mat'
# matpath = "F:/ScientificData/pencil-beam-positions/01-Jun-2022-170um_aligned_to_galvos/raylearn_pencil_beam_738673.606682_170um_aligned_to_galvos.mat"

matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-1x170um/raylearn_pencil_beam_738714.642102_1x170um.mat"

matfile = h5py.File(matpath, 'r')

cam_ft_coords_gt = (tensor((matfile['cam_ft_col'], matfile['cam_ft_row']))
    - tensor((matfile['copt_ft/Width'], matfile['copt_ft/Height'])) / 2).permute(1, 2, 0)
cam_im_coords_gt = (tensor((matfile['cam_img_col'], matfile['cam_img_row']))
    - tensor((matfile['copt_img/Width'], matfile['copt_img/Height'])) / 2).permute(1, 2, 0)

# Create TPM object and perform initial raytrace
tpm = TPM()
tpm.set_measurement(matfile)
tpm.update()
tpm.raytrace()

# Define Inital Guess
tpm.slm_zshift = tensor((0.,), requires_grad=True)
tpm.slm_angle = tensor((0.,), requires_grad=True)
tpm.galvo_roll = tensor((0.,), requires_grad=True)
tpm.cam_ft_xshift = tensor((0.,), requires_grad=True)
tpm.cam_ft_yshift = tensor((0.,), requires_grad=True)
tpm.cam_im_xshift = tensor((0.,), requires_grad=True)
tpm.cam_im_yshift = tensor((0.,), requires_grad=True)
tpm.cam_im_zshift = tensor((0.,), requires_grad=True)
tpm.sample_zshift = tensor((0.,), requires_grad=True)
tpm.obj2_zshift = tensor((0.,), requires_grad=True)
tpm.L9_zshift = tensor((0.,), requires_grad=True)
tpm.L10_zshift = tensor((0.,), requires_grad=True)
tpm.total_coverslip_thickness = tensor((170e-6,), requires_grad=True)

# Parameter groups
params = {}
params['angle'] = {
    'SLM angle': tpm.slm_angle,
    'Galvo angle': tpm.galvo_roll,
}
params['objective'] = {
    # 'OBJ2 zshift': tpm.obj2_zshift,
    # 'sample zshift': tpm.sample_zshift,
}
params['other'] = {
    # 'SLM zshift': tpm.slm_zshift,
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
        {'lr': 2.0e-3, 'params': params['angle'].values()},
        {'lr': 2.0e-4, 'params': params['objective'].values()},
        {'lr': 1.0e-3, 'params': params['other'].values()},
    ], lr=1.0e-5)

iterations = 150
errors = torch.zeros(iterations)


# Initialize logs for tracking each parameter
params_logs = {}
for groupname in params:
    params_logs[groupname] = {}
    for paramname in params[groupname]:
        params_logs[groupname][paramname] = torch.zeros(iterations)


trange = tqdm(range(iterations), desc='error: -')

# Plot
if do_plot_empty:
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

    if t % 50 == 0 and do_plot_empty:
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


if do_plot_empty:
    # Plot error
    fig, ax1 = plt.subplots(figsize=(7, 7))
    fig.dpi = 144
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


if do_plot_pincushion:
    # Plot pincushion
    plt.figure()
    plt.title('Fourier Cam Pincushionness\ncoordinate separation (diff)')
    pincushdiff_ft    = components(cam_ft_coords[   :, 12:19, :].diff(dim=1).mean(dim=0).squeeze().detach())
    pincushdiff_ft_gt = components(cam_ft_coords_gt[:, 12:19, :].diff(dim=1).mean(dim=0).squeeze().detach())

    plt.plot(pincushdiff_ft[0],    '.-', label='x sim')
    plt.plot(pincushdiff_ft[1],    '.-', label='y sim')
    plt.plot(pincushdiff_ft_gt[0], '.-', label='x measured')
    plt.plot(pincushdiff_ft_gt[1], '.-', label='y measured')
    plt.xlabel('relative index')
    plt.ylabel('x|y (pixels)')
    plt.legend()


    plt.figure()
    plt.title('Image Cam Pincushionness\ncoordinate separation (diff)')
    pincushdiff_im    = components(-cam_im_coords[4:9, :, :].diff(dim=0).mean(dim=1).squeeze().detach())
    pincushdiff_im_gt = components(-cam_im_coords_gt[4:9, :, :].diff(dim=0).mean(dim=1).squeeze().detach())

    plt.plot(pincushdiff_im[0],    '.-', label='x sim')
    plt.plot(pincushdiff_im[1],    '.-', label='y sim')
    plt.plot(pincushdiff_im_gt[0], '.-', label='x measured')
    plt.plot(pincushdiff_im_gt[1], '.-', label='y measured')
    plt.xlabel('relative index')
    plt.ylabel('x|y (pixels)')
    plt.legend()
    plt.show()




# === Glass plate === #

# Import measurement
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-400um/raylearn_pencil_beam_738477.768870_400um.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-empty/raylearn_pencil_beam_738477.729080_empty.mat'
# matpath = "F:/ScientificData/pencil-beam-positions/01-Jun-2022-170um+400um_aligned_to_galvos/raylearn_pencil_beam_738673.636565_170um+400um_aligned_to_galvos.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/01-Jun-2022-400um_aligned_to_slm/raylearn_pencil_beam_738673.696695_400um_aligned_to_slm.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/01-Jun-2022-170um+400um_aligned_to_galvos/raylearn_pencil_beam_738673.636565_170um+400um_aligned_to_galvos.mat"

# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-1x170um/raylearn_pencil_beam_738714.642102_1x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-1x170um_2nd/raylearn_pencil_beam_738714.744282_1x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-2x170um/raylearn_pencil_beam_738714.652131_2x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-2x170um_2nd/raylearn_pencil_beam_738714.757416_2x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-3x170um/raylearn_pencil_beam_738714.670969_3x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-3x170um_2nd/raylearn_pencil_beam_738714.768351_3x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-4x170um/raylearn_pencil_beam_738714.683205_4x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-4x170um_2nd/raylearn_pencil_beam_738714.784681_4x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-5x170um/raylearn_pencil_beam_738714.692862_5x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-5x170um_2nd/raylearn_pencil_beam_738714.793049_5x170um_2nd.mat"
matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-1x400um/raylearn_pencil_beam_738714.801627_1x400um.mat"

matfile = h5py.File(matpath, 'r')

cam_ft_coords_gt = (tensor((matfile['cam_ft_col'], matfile['cam_ft_row']))
    - tensor((matfile['copt_ft/Width'], matfile['copt_ft/Height'])) / 2).permute(1, 2, 0)
cam_im_coords_gt = (tensor((matfile['cam_img_col'], matfile['cam_img_row']))
    - tensor((matfile['copt_img/Width'], matfile['copt_img/Height'])) / 2).permute(1, 2, 0)

# Parameters
tpm.total_coverslip_thickness = tensor((400e-6,), requires_grad=True)
tpm.coverslip_tilt_around_x = tensor((0.0,), requires_grad=True)
tpm.coverslip_tilt_around_y = tensor((0.0,), requires_grad=True)

tpm.set_measurement(matfile)
tpm.update()

# Parameter groups
params_obj1_zshift = {}
params_obj1_zshift['angle'] = {
    # 'Coverslip tilt around x': tpm.coverslip_tilt_around_x,
    # 'Coverslip tilt around y': tpm.coverslip_tilt_around_y,
}
params_obj1_zshift['other'] = {
    'Total Coverslip Thickness': tpm.total_coverslip_thickness,
    # 'OBJ2 zshift': tpm.obj2_zshift,
    'cam im xshift': tpm.cam_im_xshift,
    'cam im yshift': tpm.cam_im_yshift,
}

# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 1.0e-2, 'params': params_obj1_zshift['angle'].values()},
        {'lr': 1.0e-4, 'params': params_obj1_zshift['other'].values()},
    ], lr=1.0e-5)

iterations = 150
errors = torch.zeros(iterations)


# Initialize logs for tracking each parameter
params_obj1_zshift_logs = {}
for groupname in params_obj1_zshift:
    params_obj1_zshift_logs[groupname] = {}
    for paramname in params_obj1_zshift[groupname]:
        params_obj1_zshift_logs[groupname][paramname] = torch.zeros(iterations)

trange = tqdm(range(iterations), desc='error: -')


# Plot
if do_plot_coverslip:
    fig, ax = plt.subplots(nrows=2, figsize=(5, 10), dpi=110)

    fig_tpm = plt.figure(figsize=(15, 4), dpi=110)
    ax_tpm = plt.gca()


for t in trange:
    # === Learn === #
    # Forward pass
    tpm.update()
    cam_ft_coords, cam_im_coords = tpm.raytrace()

    # Compute and print error
    error = 0*MSE(cam_ft_coords_gt, cam_ft_coords) \
        + MSE(cam_im_coords_gt, cam_im_coords) \

    error_value = error.detach().item()
    errors[t] = error_value

    for groupname in params_obj1_zshift:
        for paramname in params_obj1_zshift[groupname]:
            params_obj1_zshift_logs[groupname][paramname][t] = params_obj1_zshift[groupname][paramname].detach().item()

    trange.desc = f'error: {error_value:<8.3g}' \
        + f'coverslip thickness: {format_prefix(tpm.total_coverslip_thickness, "8.3f")}m'

    # error.backward(retain_graph=True)
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Plot
    if t % 100 == 0 and do_plot_coverslip:
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
        ax[0].set_title(f'Fourier Cam | coverslip={format_prefix(tpm.total_coverslip_thickness)}m | iter: {t}')

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
        tpm.plot(ax_tpm, fraction=0.1)
        plt.draw()
        plt.pause(1e-3)

for groupname in params_obj1_zshift:
    print('\n' + groupname + ':')
    for paramname in params_obj1_zshift[groupname]:
        if groupname == 'angle':
            print(f'  {paramname}: {params_obj1_zshift[groupname][paramname].detach().item():.3f}rad')
        else:
            print(f'  {paramname}: {format_prefix(params_obj1_zshift[groupname][paramname])}m')


if do_plot_coverslip:
    fig, ax1 = plt.subplots(figsize=(7, 7))
    fig.dpi = 144

    # Plot error
    errorcolor = 'tab:red'
    RMSEs = np.sqrt(errors.detach().cpu())
    ax1.plot(RMSEs, label='error', color=errorcolor)
    ax1.set_ylabel('Error (pix)')
    ax1.set_ylim((0, max(RMSEs)))
    ax1.legend(loc=2)
    ax1.legend()

    ax2 = ax1.twinx()
    for groupname in params_obj1_zshift:
        for paramname in params_obj1_zshift_logs[groupname]:
            ax2.plot(params_obj1_zshift_logs[groupname][paramname], label=paramname)
    ax2.set_ylabel('Parameter (m | rad)')
    ax2.legend(loc=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Learning parameters')
    plt.show()


# ========================================== #
# === Minimize defocus by moving z-stage === #
# Specify desired focus position
tpm.desired_focus_position_relative_to_sample_plane = tensor((0., 0., 0.))

# Parameters
tpm.obj1_zshift = tensor((0.,), requires_grad=True)

# Parameter groups
params_obj1_zshift = {}
params_obj1_zshift['angle'] = {
}
params_obj1_zshift['other'] = {
    'OBJ1 Z-shift': tpm.obj1_zshift,
}

# Create artificial 'measurement' of collimated rays from the SLM
Nx = 15
Ny = 15
x_slm = torch.tensor((1., 0.))
y_slm = torch.tensor((0., 1.))
x_array = x_slm * torch.linspace(-0.5, 0.5, Nx).view(Nx, 1, 1) * (Nx != 1)
y_array = y_slm * torch.linspace(-0.5, 0.5, Ny).view(1, Ny, 1) * (Ny != 1)
x_grid, y_grid = components(x_array + y_array)

# Mask out rays that would not fit within NA of OBJ1
radius_grid = (x_grid*x_grid + y_grid*y_grid).sqrt()
NA_radius_SLM = (tpm.f5 / tpm.f7) * tpm.fobj1 * tpm.obj1_NA / tpm.slm_height_m
mask = radius_grid < NA_radius_SLM

x_lin = x_grid[mask]
y_lin = y_grid[mask]
zero_lin = torch.zeros(x_lin.shape)

# Define 'measurement' Galvo and SLM settings
matfile = {
    'p/rects': torch.stack((x_lin, y_lin, zero_lin, zero_lin)),
    'p/galvoXs': (0.,),
    'p/galvoYs': (0.,),
    'p/GalvoXcenter': (0.,),
    'p/GalvoYcenter': (0.,)}

tpm.set_measurement(matfile)
tpm.update()


# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 1.0e-2, 'params': params_obj1_zshift['angle'].values()},
        {'lr': 1.0e-4, 'params': params_obj1_zshift['other'].values()},
    ], lr=1.0e-5)

iterations = 150
errors = torch.zeros(iterations)


# Initialize logs for tracking each parameter
params_obj1_zshift_logs = {}
for groupname in params_obj1_zshift:
    params_obj1_zshift_logs[groupname] = {}
    for paramname in params_obj1_zshift[groupname]:
        params_obj1_zshift_logs[groupname][paramname] = torch.zeros(iterations)

trange = tqdm(range(iterations), desc='error: -')


# Plot
if do_plot_obj1_zshift:
    fig_tpm = plt.figure(figsize=(15, 4), dpi=110)
    ax_tpm = plt.gca()


for t in trange:
    # === Learn === #
    # Forward pass
    tpm.update()
    cam_ft_coords, cam_im_coords = tpm.raytrace()

    # Compute and print error
    rays_focus = tpm.rays_on_desired_focus_plane
    error = MSE(rays_focus.position_m, tpm.desired_focus_plane.position_m)

    error_value = error.detach().item()
    errors[t] = error_value

    for groupname in params_obj1_zshift:
        for paramname in params_obj1_zshift[groupname]:
            params_obj1_zshift_logs[groupname][paramname][t] = params_obj1_zshift[groupname][paramname].detach().item()

    trange.desc = f'error: {error_value:<8.3g}' \
        + f'OBJ1 z-shift: {format_prefix(tpm.obj1_zshift, "8.3f")}m'

    # error.backward(retain_graph=True)
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Plot
    if t % 20 == 0 and do_plot_obj1_zshift:
        plt.figure(fig_tpm.number)
        ax_tpm.clear()
        tpm.plot(ax_tpm, fraction=0.2)
        plt.draw()
        plt.pause(1e-3)

for groupname in params_obj1_zshift:
    print('\n' + groupname + ':')
    for paramname in params_obj1_zshift[groupname]:
        if groupname == 'angle':
            print(f'  {paramname}: {params_obj1_zshift[groupname][paramname].detach().item():.3f}rad')
        else:
            print(f'  {paramname}: {format_prefix(params_obj1_zshift[groupname][paramname])}m')


if do_plot_obj1_zshift:
    fig, ax1 = plt.subplots(figsize=(7, 7))
    fig.dpi = 144

    # Plot error
    errorcolor = 'tab:red'
    RMSEs = np.sqrt(errors.detach().cpu())
    ax1.plot(RMSEs, label='error', color=errorcolor)
    ax1.set_ylabel('Error (pix)')
    ax1.set_ylim((0, max(RMSEs)))
    ax1.legend(loc=2)
    ax1.legend()

    ax2 = ax1.twinx()
    for groupname in params_obj1_zshift:
        for paramname in params_obj1_zshift_logs[groupname]:
            ax2.plot(params_obj1_zshift_logs[groupname][paramname], label=paramname)
    ax2.set_ylabel('Parameter (m | rad)')
    ax2.legend(loc=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Learning parameters')
    plt.show()


# Place point source at focal position
# Choose the opening angle such that the NA is overfilled
tpm.backtrace_source_opening_tan_angle = 1.5 * np.tan(np.arcsin(tpm.obj1_NA / tpm.n_coverslip))
tpm.update()
backrays_at_slm = tpm.backtrace()

# Propagate to screen and compute screen coordinates
coords = tpm.slm_plane.transform_rays(backrays_at_slm).detach()

wavelength_m = 804e-9

# Compute field with interpolate_shader
pathlength_to_slm = backrays_at_slm.pathlength_m.detach()
data = torch.cat((coords, pathlength_to_slm), 2)
extent = (-0.5, 0.5, -0.5, 0.5)
field_out = torch.tensor(interpolate_shader(
    data.detach().numpy(),
    npoints=(tpm.slm_height_pixels, tpm.slm_height_pixels),
    limits=extent,
    wavelength_m=wavelength_m,
    ))


plt.imshow(np.angle(field_out), extent=extent)
plt.title('SLM phase correction pattern, '
          + f'λ={format_prefix(wavelength_m, formatspec=".0f")}m'
          + f'\ncoverslip={format_prefix(tpm.total_coverslip_thickness, formatspec=".0f")}m,'
          + f' defocus={format_prefix(tpm.obj1_zshift, formatspec=".0f")}m')
plt.xlabel('x (SLM heights)')
plt.ylabel('y (SLM heights)')
plt.colorbar()
plt.show()
