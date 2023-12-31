"""
Learn the parameters of a tube in the Two Photon Microscope.
"""

import torch
from torch import tensor
import numpy as np
import h5py
import hdf5storage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
# from torchviz import make_dot

from plot_functions import plot_coords, format_prefix, plot_rays, plot_lens, plot_plane, default_viewplane
from testing import MSE, SSE, weighted_SSE
from tpm import TPM
from vector_functions import components, rejection
from interpolate_shader import interpolate_shader
from optical import Coverslip
from sample_tube import SampleTube
from ray_plane import CoordPlane, detach
from math_functions import pyramid
from dirconfig_raylearn import dirs


# Set default tensor type to double (64 bit)
# Machine epsilon of float (32-bit) is 2^-23 = 1.19e-7
# The ray simulation contains both meter and micrometer scales,
# hence floats might not be precise enough.
# https://en.wikipedia.org/wiki/Machine_epsilon
torch.set_default_tensor_type('torch.DoubleTensor')

do_plot_empty = False
do_save_frames_empty = True
do_plot_pincushion = False
do_plot_tube = True
do_plot_obj1_zshift = False
do_plot_backtrace = False
do_plot_phase_pattern = True

plt.rc('font', size=12)

# Import measurement
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-400um/raylearn_pencil_beam_738477.786123_400um.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-empty/raylearn_pencil_beam_738477.729080_empty.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/10-Feb-2022-empty/raylearn_pencil_beam_738562.645439_empty.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/01-Jun-2022-170um_aligned_to_galvos/raylearn_pencil_beam_738673.606682_170um_aligned_to_galvos.mat'
# matpath = "F:/ScientificData/pencil-beam-positions/01-Jun-2022-170um_aligned_to_galvos/raylearn_pencil_beam_738673.606682_170um_aligned_to_galvos.mat"

# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-1x170um/raylearn_pencil_beam_738714.642102_1x170um.mat"


# matpath = dirs['localdata'].joinpath("raylearn-data/TPM/pencil-beam-positions/30-Sep-2022-1x170um/raylearn_pencil_beam_738794.634384_1x170um.mat")

matpath = dirs['localdata'].joinpath("raylearn-data/TPM/pencil-beam-positions/10-Mar-2023-coverslip170um/raylearn_pencil_beam_738955.694554_coverslip170um.mat")

matfile = h5py.File(matpath, 'r')

cam_ft_coords_gt = (tensor((matfile['cam_ft_col'], matfile['cam_ft_row']))
    - tensor((matfile['copt_ft/Width'], matfile['copt_ft/Height'])) / 2).permute(1, 2, 0)
cam_im_coords_gt = (tensor((matfile['cam_img_col'], matfile['cam_img_row']))
    - tensor((matfile['copt_img/Width'], matfile['copt_img/Height'])) / 2).permute(1, 2, 0)

# Create TPM object and perform initial raytrace
tpm = TPM()
tpm.set_measurement(matfile)
tpm.sample = Coverslip()
tpm.update()
tpm.sample.coverslip_front_plane = tpm.sample_plane
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
tpm.galvo_volts_per_optical_degree.requires_grad=True

# Parameter groups
params = {}
params['angle'] = {
    'SLM angle (rad)': tpm.slm_angle,
    'Galvo angle (rad)': tpm.galvo_roll,
}
params['objective'] = {
    'OBJ2 zshift (m)': tpm.obj2_zshift,
    # 'sample zshift (m)': tpm.sample_zshift,
}
params['other'] = {
    # 'SLM zshift (m)': tpm.slm_zshift,
    # 'L9 zshift (m)': tpm.L9_zshift,
    'cam ft xshift (m)': tpm.cam_ft_xshift,
    'cam ft yshift (m)': tpm.cam_ft_yshift,
    'cam im xshift (m)': tpm.cam_im_xshift,
    'cam im yshift (m)': tpm.cam_im_yshift,
    # 'cam im zshift (m)': tpm.cam_im_zshift,
    # 'Galvo response (V/deg)': tpm.galvo_volts_per_optical_degree,
}

tpm.update()

# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 2.0e-3, 'params': params['angle'].values()},
        {'lr': 2.0e-4, 'params': params['objective'].values()},
        {'lr': 1.0e-3, 'params': params['other'].values()},
    ], lr=1.0e-5, amsgrad=True)

iterations = 150
errors = torch.zeros(iterations) * torch.nan


# Initialize logs for tracking each parameter
params_logs = {}
for groupname in params:
    params_logs[groupname] = {}
    for paramname in params[groupname]:
        params_logs[groupname][paramname] = torch.nan * torch.zeros(iterations)


trange = tqdm(range(iterations), desc='error: -')

        # Parameters
        # tpm.total_coverslip_thickness = tensor((1170e-6,), requires_grad=True)

        # Trace computational graph
        # tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})


# Plot
if do_plot_empty:
    fig, ax = plt.subplots(nrows=2, figsize=(5, 10), dpi=110)

    fig_tpm = plt.figure(figsize=(15, 4), dpi=110)
    ax_tpm = plt.gca()

if do_save_frames_empty:
    fig_save_frames_empty = plt.figure(dpi=100, figsize=(11, 7))

    gs = fig_save_frames_empty.add_gridspec(2, 3)
    gs.hspace = 0.3
    ax_ray = fig_save_frames_empty.add_subplot(gs[0, :-1])
    ax_ft = fig_save_frames_empty.add_subplot(gs[1, 0])
    ax_im = fig_save_frames_empty.add_subplot(gs[1, 1])
    ax_err = fig_save_frames_empty.add_subplot(gs[:, 2])


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

    if t % 10 == 0 and do_plot_empty:
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
        plt.pause(1e-1)

    if do_save_frames_empty and t % 30 == 0:
        plt.figure(fig_save_frames_empty.number)

        # Fourier cam
        ax_ft.clear()
        cam_ft_coord_pairs_x, cam_ft_coord_pairs_y = \
            torch.stack((cam_ft_coords_gt, cam_ft_coords)).detach().unbind(-1)

        ax_ft.plot(cam_ft_coord_pairs_x.view(2, -1), cam_ft_coord_pairs_y.view(2, -1),
                   color='lightgrey')
        plot_coords(ax_ft, cam_ft_coords_gt[:, :, :], {'label': 'measured'})
        plot_coords(ax_ft, cam_ft_coords[:, :, :], {'label': 'simulated'})

        ax_ft.set_ylabel('y (pix)')
        ax_ft.set_xlabel('x (pix)')
        ax_ft.legend(loc=1)
        ax_ft.set_title('Fourier Cam')
        ax_ft.set_xlim((-370, 250))
        ax_ft.set_ylim((-250, 370))
        ax_ft.set_aspect(1)

        # Image cam
        cam_im_coord_pairs_x, cam_im_coord_pairs_y = \
            torch.stack((cam_im_coords_gt, cam_im_coords)).detach().unbind(-1)

        ax_im.clear()
        ax_im.plot(cam_im_coord_pairs_x.view(2, -1), cam_im_coord_pairs_y.view(2, -1),
                   color='lightgrey')
        plot_coords(ax_im, cam_im_coords_gt[:, :, :], {'label': 'measured'})
        plot_coords(ax_im, cam_im_coords[:, :, :], {'label': 'simulated'})

        ax_im.set_ylabel('y (pix)')
        ax_im.set_xlabel('x (pix)')
        ax_im.legend(loc=1)
        ax_im.set_title('Image Cam')
        ax_im.set_xlim((-500, 500))
        ax_im.set_ylim((-600, 600))
        ax_im.set_aspect(1)

        # Plot error and parameters
        ax_err.clear()
        errorcolor = 'darkred'
        MSEs = errors.detach().cpu() * 1e-7
        ax_err.plot(MSEs, label='error (a.u.)', color=errorcolor, linewidth=2.5)
        ax_err.set_xlabel('iteration')
        ax_err.set_ylim((-6e-3, 6e-3))
        ax_err.set_xlim((0, iterations))

        for groupname in params:
            for paramname in params_logs[groupname]:
                ax_err.plot(params_logs[groupname][paramname], label=paramname)
        ax_err.legend(loc=1)
        ax_err.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax_ray.clear()
        tpm.plot(ax_ray)

        plt.draw()
        plt.pause(1e-3)

        # Uncomment to save the frames
        # plt.savefig(str(dirs['localdata'].joinpath(f'plots/learn-calibration-it{t:03d}.png')))



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




# === Sample === #

# Import measurement
matpath = dirs["localdata"].joinpath("raylearn-data/TPM/pencil-beam-positions/11-Mar-2023-tube-0/raylearn_pencil_beam_738956.722609_tube-0.5uL.mat")

matfile = h5py.File(matpath, 'r')

# Detected beam spot coordinates in pix
cam_ft_coords_gt = (tensor((matfile['cam_ft_col'], matfile['cam_ft_row']))
    - tensor((matfile['copt_ft/Width'], matfile['copt_ft/Height'])) / 2).permute(1, 2, 0)
cam_im_coords_gt = (tensor((matfile['cam_img_col'], matfile['cam_img_row']))
    - tensor((matfile['copt_img/Width'], matfile['copt_img/Height'])) / 2).permute(1, 2, 0)

# Discard coords marked as spot not found
found_spot_ft = tensor(matfile['found_spot_ft'])
found_spot_im = tensor(matfile['found_spot_img'])
found_spot_coords_ft = torch.stack((found_spot_ft, found_spot_ft), dim=2)
found_spot_coords_im = torch.stack((found_spot_im, found_spot_im), dim=2)

# Weight measured coordinates
# Use 1 / (detected beam spot standard deviation in pix) as weight
# Coordinates are also in pix, so division makes result unitless
# If the spot can't be found, the values are nan. In those cases the inverse std is set to 0
invstd_cam_ft_coords = (1 / (tensor((matfile['cam_ft_col_std'], matfile['cam_ft_row_std']))).permute(1, 2, 0)).nan_to_num(0.0)
invstd_cam_im_coords = (1 / (tensor((matfile['cam_img_col_std'], matfile['cam_img_row_std']))).permute(1, 2, 0)).nan_to_num(0.0)      ########### some are nan


# Initial conditions
########### Base these on values optimized for initial guess? Only at first? Or dynamically?
shell_thickness_m_init = (610e-6 - 143e-6) / 2
outer_radius_m_init = 610e-6 / 2
sample_zshift_init = 260e-6
obj2_zshift_init = 250e-6
tube_angle_init = np.radians(90.)

weight_shell_thickness_m_init = 1 / (0.01 * shell_thickness_m_init)  # According to AlphaLabs: volume accuracy rating ±1%
weight_outer_radius_m_init = 1 / 20e-6          # Mitutoyo Absolute Digimatic Calipers accuracy
weight_sample_zshift_init = 1 / 200e-6
weight_obj2_zshift_init = 1 / 200e-6
weight_tube_angle_init = 1 / np.radians(5.0)    # Estimated accuracy of tube alignment

# Parameters
# tpm.total_coverslip_thickness = tensor((1170e-6,), requires_grad=True)

tpm.set_measurement(matfile)
tpm.sample = SampleTube()
tpm.sample.tube_angle = tensor((tube_angle_init,), requires_grad=True)

tpm.sample.shell_thickness_m = tensor((shell_thickness_m_init), requires_grad=True)
tpm.sample.outer_radius_m = tensor((outer_radius_m_init,), requires_grad=True)
tpm.sample_zshift = tensor((sample_zshift_init,), requires_grad=True)
tpm.obj2_xshift = tensor((0.,), requires_grad=True)
tpm.obj2_yshift = tensor((0.,), requires_grad=True)
tpm.obj2_zshift = tensor((obj2_zshift_init,), requires_grad=True)

tpm.backtrace_Nx = 21
tpm.backtrace_Ny = 21

tpm.update()

# Parameter groups
params_obj1_zshift = {}
params_obj1_zshift['angle'] = {
    # 'Coverslip tilt around x': tpm.coverslip_tilt_around_x,
    # 'Coverslip tilt around y': tpm.coverslip_tilt_around_y,
    'Tube angle': tpm.sample.tube_angle
}
params_obj1_zshift['obj'] = {
    'Sample Plane z-shift': tpm.sample_zshift,
    'OBJ2 x-shift': tpm.obj2_xshift,
    'OBJ2 y-shift': tpm.obj2_yshift,
    'OBJ2 z-shift': tpm.obj2_zshift,

}
params_obj1_zshift['other'] = {
    # 'Total Coverslip Thickness': tpm.total_coverslip_thickness,
    'cam im xshift': tpm.cam_im_xshift,
    'cam im yshift': tpm.cam_im_yshift,
    'shell thickness': tpm.sample.shell_thickness_m,
    'outer radius': tpm.sample.outer_radius_m,
}

# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 2.0e-2, 'params': params_obj1_zshift['angle'].values()},
        {'lr': 1.0e-5, 'params': params_obj1_zshift['obj'].values()},
        {'lr': 1.0e-6, 'params': params_obj1_zshift['other'].values()},
    ], lr=1.0e-5, amsgrad=True)

iterations = 1000
errors = torch.zeros(iterations)


# Initialize logs for tracking each parameter
params_obj1_zshift_logs = {}
for groupname in params_obj1_zshift:
    params_obj1_zshift_logs[groupname] = {}
    for paramname in params_obj1_zshift[groupname]:
        params_obj1_zshift_logs[groupname][paramname] = torch.zeros(iterations)

trange = tqdm(range(iterations), desc='error: -')


# Plot
if do_plot_tube:
    fig, ax = plt.subplots(nrows=2, figsize=(5, 10), dpi=110)

    fig_tpm = plt.figure(figsize=(7, 7), dpi=110)
    ax_tpm = plt.gca()


torch.autograd.set_detect_anomaly(True)         ############## Use to detect NaNs in backward pass

nans = []                                       # Count NaNs

for t in trange:
    # === Learn sample === #
    # Forward pass
    tpm.update()
    cam_ft_coords, cam_im_coords = tpm.raytrace()

    # Compute error

    # Ignore all values that are marked as undetected or yield NaN in computation
    # mask_nanless = torch.logical_or(torch.logical_or(cam_ft_coords_gt.isnan(), cam_ft_coords.isnan()), torch.logical_or(cam_im_coords_gt.isnan(), cam_ft_coords.isnan())).logical_not()

    # Create masks to find different cases
    cam_ft_size = tensor((matfile['/copt_ft/Width'], matfile['/copt_ft/Height'])).view(1, 1, 2)
    cam_im_size = tensor((matfile['/copt_img/Width'], matfile['/copt_img/Height'])).view(1, 1, 2)
    ft_raytrace_inside = (cam_ft_coords.abs() < (cam_ft_size / 2)).prod(dim=-1)
    im_raytrace_inside = (cam_im_coords.abs() < (cam_im_size / 2)).prod(dim=-1)

    weight_cam_ft_coords = invstd_cam_ft_coords * tpm.cam_ft_ray.intensity
    weight_cam_im_coords = invstd_cam_im_coords * tpm.cam_im_ray.intensity      ######### some invstd_cam_im_coords are nan, since those spots aren't detected

    # Error with measurements
    error_measure = \
        weighted_SSE(cam_ft_coords_gt[found_spot_coords_ft],            # Fourier cam coords
                     cam_ft_coords[found_spot_coords_ft],
                     weight_cam_ft_coords[found_spot_coords_ft]) + \
        weighted_SSE(cam_im_coords_gt[found_spot_coords_im],            # Image cam coords
                     cam_im_coords[found_spot_coords_im],
                     weight_cam_im_coords[found_spot_coords_im])

    # Error of undetected measurements, but ray traced spot is on camera
    error_edge = \
        weighted_SSE(0, pyramid(cam_ft_size, cam_ft_coords)[found_spot_coords_ft.logical_not()],
        weight_cam_ft_coords[found_spot_coords_ft.logical_not()]) + \
        weighted_SSE(0, pyramid(cam_im_size, cam_im_coords)[found_spot_coords_im.logical_not()],
        weight_cam_im_coords[found_spot_coords_im.logical_not()])       ####### Some are inf

    # Error with initial guess
    error_init = \
        weighted_SSE(tpm.sample.shell_thickness_m,              # Shell thickness
                     shell_thickness_m_init,
                     weight_shell_thickness_m_init) + \
        weighted_SSE(tpm.sample.outer_radius_m,                 # Outer radius
                     outer_radius_m_init,
                     weight_outer_radius_m_init) + \
        weighted_SSE(tpm.sample_zshift,                         # Sample z-shift
                     sample_zshift_init,
                     weight_sample_zshift_init) + \
        weighted_SSE(tpm.obj2_zshift,                           # OBJ2 z-shift
                     obj2_zshift_init,
                     weight_obj2_zshift_init) + \
        weighted_SSE(tpm.sample.tube_angle,                     # Tube angle
                     tube_angle_init,
                     weight_tube_angle_init)

    # Error intensity   ############## SSE
    error_intense = \
        (found_spot_ft - tpm.cam_ft_ray.intensity.squeeze()*ft_raytrace_inside).relu().square().sum() + \
        (found_spot_im - tpm.cam_im_ray.intensity.squeeze()*im_raytrace_inside).relu().square().sum()

    # Total error
    error = error_measure + error_init + error_edge + error_intense

    # Count NaN occurrences in ray tracing output and add to list
    nans += [cam_ft_coords.isnan().sum()]

    # # Uncomment to search namespace for tensor objects containing NaNs
    # import gc
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             # print(type(obj), obj.size())
    #             if obj.isnan().any():
    #                 print(type(obj), obj.size())
    #                 pass
    #     except:
    #         pass

    # init_fraction = (error_init / error).detach().item()

    error_value = error.detach().item()
    errors[t] = error_value

    for groupname in params_obj1_zshift:
        for paramname in params_obj1_zshift[groupname]:
            params_obj1_zshift_logs[groupname][paramname][t] = params_obj1_zshift[groupname][paramname].detach().item()

    trange.desc = f'error: {error_value:<8.3g}' \
        # + f' init fraction: {init_fraction:.3f}'

    # Plot
    if t % 1 == 0 and do_plot_tube and t >= 0:
        plt.figure(fig.number)

        # for n in range(52):
        for n in [35]:

            # Fourier cam
            cam_ft_coord_pairs_x, cam_ft_coord_pairs_y = \
                torch.stack((cam_ft_coords_gt, cam_ft_coords)).detach().unbind(-1)

            ax[0].clear()
            # ax[0].plot(cam_ft_coord_pairs_x.view(2, -1), cam_ft_coord_pairs_y.view(2, -1),
            #         color='lightgrey')
            plot_coords(ax[0], cam_ft_coords_gt[:, :, :], {'label': 'measured'})
            plot_coords(ax[0], cam_ft_coords[:, :, :], {'label': 'sim'})
            x_ft, y_ft = cam_ft_coords[15, n, :].detach().unbind(-1)
            ax[0].plot(x_ft, y_ft, 'ok', markersize=22, fillstyle='none', label=f'angle #{n}')

            ax[0].set_ylabel('y (pix)')
            ax[0].set_xlabel('x (pix)')
            ax[0].legend(loc=1)
            ax[0].set_title(f'Fourier Cam | iter: {t}')

            # Image cam
            cam_im_coord_pairs_x, cam_im_coord_pairs_y = \
                torch.stack((cam_im_coords_gt, cam_im_coords)).detach().unbind(-1)

            ax[1].clear()
            ax[1].plot(544 * tensor((1, 1, -1, -1, 1.)), 544 * tensor((1, -1, -1, 1, 1.)), '-k', label='cam edge')
            # ax[1].plot(cam_im_coord_pairs_x.view(2, -1), cam_im_coord_pairs_y.view(2, -1),
            #            color='lightgrey')
            plot_coords(ax[1], cam_im_coords_gt[:, n, :], {'label': 'measured'})
            plot_coords(ax[1], cam_im_coords[:, n, :], {'label': 'sim', 'markersize': 1.5})

            ax[1].set_ylabel('y (pix)')
            ax[1].set_xlabel('x (pix)')
            ax[1].legend(loc=1)
            ax[1].set_title(f'Image Cam | iter: {t} | angle #{n}')

            ######
            ax[1].set_xlim((-600, 600))
            ax[1].set_ylim((-600, 600))
            ######

            plt.draw()
            # plt.savefig(f'/home/dani/raylearn/plots/tube_spot_positions/cam_spots_angle{n:02}.png', bbox_inches='tight')
            plt.pause(1e-3)

        plt.figure(fig_tpm.number)
        ax_tpm.clear()
        # viewplane = default_viewplane()
        viewplane = detach(tpm.sample.cyl_plane)

        tpm.plot(ax_tpm, fraction=0.1, viewplane=viewplane)
        # tpm.sample.plot(ax_tpm)
        # x_sample, y_sample = viewplane.transform_points(tpm.sample.slide_top_plane.position_m)
        # ax_tpm.set_xlim((x_sample - 3 * tpm.sample.outer_radius_m).detach(), (x_sample + 1.5*tpm.sample.slide_thickness_m).detach())
        # ax_tpm.set_ylim((y_sample - 2 * tpm.sample.outer_radius_m).detach(), (y_sample + 2 * tpm.sample.outer_radius_m).detach())
        ax_tpm.set_xlim((-0.6e-3, 0.6e-3))
        ax_tpm.set_ylim((-700e-6, 500e-6))
        ax_tpm.set_aspect(1)

        plt.draw()
        plt.pause(1e-3)


    # error.backward(create_graph=True)
    # error.backward(retain_graph=True)
    error.backward()

    # # Uncomment to walk the computational graph, searching for a specified function name
    # # Note: retain_graph=True is required for the backwards pass
    # def walk_graph(node, depth):
    #     # print(depth)
    #     if not(node is None):
    #         if node.name() == 'PowBackward0':
    #             print(depth, node._saved_self)
    #         for next_func in node.next_functions:
    #             walk_graph(next_func[0], depth+1)

    # walk_graph(error.grad_fn, 0)

    optimizer.step()
    optimizer.zero_grad()

for groupname in params_obj1_zshift:
    print('\n' + groupname + ':')
    for paramname in params_obj1_zshift[groupname]:
        if groupname == 'angle':
            print(f'  {paramname}: {params_obj1_zshift[groupname][paramname].detach().item():.3f}rad')
        else:
            print(f'  {paramname}: {format_prefix(params_obj1_zshift[groupname][paramname], ".3f")}m')


if do_plot_tube and iterations > 0:
    fig, ax1 = plt.subplots(figsize=(8, 8))
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
tpm.obj1_zshift.requires_grad = True
tpm.update()

# Parameter groups
params_obj1_zshift = {}
params_obj1_zshift['angle'] = {
}
params_obj1_zshift['other'] = {
    'OBJ1 Z-shift': tpm.obj1_zshift,
}

# Create artificial 'measurement' of collimated rays from the SLM
Nx = 21
Ny = 21
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
    ], lr=1.0e-5, amsgrad=True)

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
    # === Learn z-shift === #
    # Forward pass
    tpm.backtrace_Nx = 300
    tpm.backtrace_Ny = 300
    tpm.update()
    cam_ft_coords, cam_im_coords = tpm.raytrace()

    # Compute and print error
    rays_focus = tpm.sample.rays_at_desired_focus
    error = MSE(rays_focus.position_m, tpm.sample.desired_focus_plane.position_m)

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


################# Use inference mode
# Place point source at focal position
# Choose the opening angle such that the NA is overfilled
tpm.sample.n_tube = 1.5106          # Schott N-BK7 @808nm
tpm.sample.n_slide = 1.5170         # Soda Lime Glass @808nm
tpm.sample.n_inside = 1.3290        # Water inside the tube @808nm
tpm.sample.n_outside = 1.3290       # Water between tube and slide @808nm
tpm.backtrace_source_opening_tan_angle = 1.2 * np.tan(np.arcsin(tpm.obj1_NA / tpm.n_water))
tpm.update()
backrays_at_slm = tpm.backtrace()

# Propagate to screen and compute screen coordinates
coords = tpm.slm_plane.transform_rays(backrays_at_slm).detach()

wavelength_m = 808e-9

# Compute field with interpolate_shader
pathlength_to_slm = backrays_at_slm.pathlength_m.detach()
data = torch.cat((coords, pathlength_to_slm), 2)
a = 0.5
# a = NA_radius_SLM
extent = (-a, a, -a, a)
field_SLM = torch.tensor(interpolate_shader(
    data.detach().numpy(),
    npoints=(tpm.slm_height_pixels, tpm.slm_height_pixels),
    limits=extent,
    wavelength_m=wavelength_m,
    ))


if do_plot_backtrace:
    # Plot rays of backward raytrace
    plt.figure()
    ax = plt.gca()
    scale = 0.02
    plot_plane(ax, tpm.slm_plane, 0.8, ' SLM', plotkwargs={'color': 'red'})
    plot_lens(ax, tpm.L5, tpm.f5, scale, ' L5\n')
    plot_lens(ax, tpm.L7, tpm.f7, scale, ' L7\n')
    plot_lens(ax, tpm.OBJ1, tpm.fobj1, scale, ' OBJ1\n')
    # plot_plane(ax, tpm.sample_plane, scale*0.8, '', ' sample plane')
    plot_plane(ax, tpm.desired_focus_plane, scale * 0.07, ' Desired\n Focus\n Plane')

    plot_rays(ax, tpm.backrays, fraction=0.0005)
    tpm.sample.plot(ax, plotkwargs={'color': 'green'})

    ax.set_xlabel('optical axis, z (m)')
    ax.set_ylabel('y (m)')

    plt.show()


if do_plot_phase_pattern:
    # Plot SLM phase correction pattern
    fig_phase_pattern = plt.figure(dpi=200, figsize=(6, 5.5))
    plt.imshow(np.angle(field_SLM), extent=extent, vmin=-np.pi, vmax=np.pi, cmap='twilight', interpolation='nearest')
    plt.title(f'Phase correction pattern, '
              + f'λ={format_prefix(wavelength_m)}m'
              + f'\nshell thickness={format_prefix(tpm.sample.shell_thickness_m)}m,'
              + f' outer radius={format_prefix(tpm.sample.outer_radius_m)}m')

    # Draw a circle to indicate NA
    N_verts_NA_circle = 200
    theta_vert_NA_circle = np.linspace(0, 2*np.pi, N_verts_NA_circle)
    x_NA_circle = NA_radius_SLM * np.cos(theta_vert_NA_circle)
    y_NA_circle = NA_radius_SLM * np.sin(theta_vert_NA_circle)
    plt.plot(x_NA_circle, y_NA_circle, '--g', linewidth=2.5, label=f'NA={tpm.obj1_NA}')
    plt.legend(loc='upper right')

    plt.xticks(())
    plt.yticks(())

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)      # Colobar Axes
    colorbar = plt.colorbar(ticks=[-np.pi, 0, np.pi], label='Phase', cax=cax)
    colorbar.ax.set_yticklabels(['−π', '0', '+π'])
    plt.show()

# Save file
matpath_out = str(dirs['localdata'].joinpath('raylearn-data/TPM/pattern-' \
    + '0.5uL-tube-' \
    + f'λ{format_prefix(wavelength_m, formatspec=".1f")}m.mat'))
mdict = {
    'matpath_pencil_beam_positions': str(matpath),
    'field_SLM': field_SLM.detach().numpy(),
    'obj1_zshift': tpm.obj1_zshift.detach().numpy(),
    'inner_radius_m': tpm.sample.inner_radius_m.detach().numpy(),
    'outer_radius_m': tpm.sample.outer_radius_m.detach().numpy(), ########### add more properties
    'shell_thickness_m': tpm.sample.shell_thickness_m.detach().numpy(),
    'wavelength_m': wavelength_m,
}
matfile_out = hdf5storage.savemat(matpath_out, mdict)
print(f'Saved to {matpath_out}')
