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

do_plot_tube = True

# Create synthetic version

# Define 'measurement' Galvo and SLM settings
matfile = {
    'p/rects': 0.4 * torch.tensor(((0., 1, 0, 0), (-1, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, -1, 0, 0))).T,
    'p/galvoXs': ((0, -0.04, 0, 0.04, 0),),
    'p/galvoYs': ((0.04, 0, 0, 0, -0.04),),
    'p/GalvoXcenter': (0.,),
    'p/GalvoYcenter': (0.,)}

tpm = TPM()

# Ground truth
tpm.set_measurement(matfile)
tpm.sample = SampleTube()
tpm.sample.tube_angle = tensor((np.radians(90.),))
tpm.sample_zshift = tensor((150e-6,))
tpm.obj2_zshift = tensor((310e-6,))

tpm.update()
cam_ft_coords_synth_gt, cam_im_coords_synth_gt = tpm.raytrace()


# # Plot ground truth raytracing
# ax_tpm = plt.gca()
# tpm.plot(ax_tpm)
# viewplane = default_viewplane()
# x_sample, y_sample = viewplane.transform_points(tpm.sample.slide_top_plane.position_m)
# ax_tpm.set_xlim((x_sample - 3 * tpm.sample.outer_radius_m).detach(), (x_sample + 1.5*tpm.sample.slide_thickness_m).detach())
# ax_tpm.set_ylim((y_sample - 2 * tpm.sample.outer_radius_m).detach(), (y_sample + 2 * tpm.sample.outer_radius_m).detach())
# ax_tpm.set_aspect(1)
# plt.show()


# noise = torch.linspace()
# trange = tqdm(range(iterations), desc='noisy synth: -')

# for t in trange:
# Initial conditions
tpm.sample.shell_thickness_m = tensor((100e-6,), requires_grad=True)
tpm.sample.outer_radius_m = tensor((450e-6,), requires_grad=True)
# tpm.sample_zshift = tensor((-50e-6,), requires_grad=True)
# tpm.obj2_zshift = tensor((0e-6,), requires_grad=True)
tpm.sample_zshift.requires_grad = True
tpm.obj2_zshift.requires_grad = True

tpm.backtrace_Nx = 21
tpm.backtrace_Ny = 21

tpm.update()

# Parameter groups
params = {}
params['angle'] = {
    # 'Tube angle': tpm.sample.tube_angle
}
params['obj'] = {
    'Sample Plane z-shift': tpm.sample_zshift,
    'OBJ2 z-shift': tpm.obj2_zshift,
}
params['other'] = {
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
        {'lr': 2.0e-2, 'params': params['angle'].values()},
        {'lr': 3.0e-5, 'params': params['obj'].values()},
        {'lr': 3.0e-5, 'params': params['other'].values()},
    ], lr=1.0e-5)

iterations = 1200
errors = torch.zeros(iterations)


# Initialize logs for tracking each parameter
params_obj1_zshift_logs = {}
for groupname in params:
    params_obj1_zshift_logs[groupname] = {}
    for paramname in params[groupname]:
        params_obj1_zshift_logs[groupname][paramname] = torch.zeros(iterations)

trange = tqdm(range(iterations), desc='error: -')


# Plot
if do_plot_tube:
    fig, ax = plt.subplots(nrows=2, figsize=(5, 10), dpi=110)

    fig_tpm = plt.figure(figsize=(5, 5), dpi=110)
    ax_tpm = plt.gca()


for t in trange:
    # === Learn sample === #
    # Forward pass
    tpm.update()
    cam_ft_coords, cam_im_coords = tpm.raytrace()
    slm_rays = tpm.backtrace()
    slm_dir_rej = rejection(slm_rays.direction, tpm.slm_plane.normal)

    # Compute and print error
    error = MSE(cam_ft_coords_synth_gt, cam_ft_coords) \
        + MSE(cam_im_coords_synth_gt, cam_im_coords) \
        # + 1e9 * MSE(slm_dir_rej, slm_dir_rej.mean())

    # print(MSE(slm_dir_rej, slm_dir_rej.mean()) / error)

    error_value = error.detach().item()
    errors[t] = error_value

    for groupname in params:
        for paramname in params[groupname]:
            params_obj1_zshift_logs[groupname][paramname][t] = params[groupname][paramname].detach().item()

    trange.desc = f'error: {error_value:<8.3g}' \
        + f'coverslip thickness: {format_prefix(tpm.total_coverslip_thickness, "8.3f")}m'

    # error.backward(retain_graph=True)
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Plot
    if t % 250 == 0 and do_plot_tube:
        plt.figure(fig.number)

        # Fourier cam
        cam_ft_coord_pairs_x, cam_ft_coord_pairs_y = \
            torch.stack((cam_ft_coords_synth_gt, cam_ft_coords)).detach().unbind(-1)

        ax[0].clear()
        ax[0].plot(cam_ft_coord_pairs_x.view(2, -1), cam_ft_coord_pairs_y.view(2, -1),
                color='lightgrey')
        plot_coords(ax[0], cam_ft_coords_synth_gt[:, :, :], {'label': 'measured'})
        plot_coords(ax[0], cam_ft_coords[:, :, :], {'label': 'sim'})

        ax[0].set_ylabel('y (pix)')
        ax[0].set_xlabel('x (pix)')
        ax[0].legend(loc=1)
        ax[0].set_title(f'Fourier Cam | coverslip={format_prefix(tpm.total_coverslip_thickness)}m | iter: {t}')

        # Image cam
        cam_im_coord_pairs_x, cam_im_coord_pairs_y = \
            torch.stack((cam_im_coords_synth_gt, cam_im_coords)).detach().unbind(-1)

        ax[1].clear()
        ax[1].plot(cam_im_coord_pairs_x.view(2, -1), cam_im_coord_pairs_y.view(2, -1),
                color='lightgrey')
        plot_coords(ax[1], cam_im_coords_synth_gt[:, :, :], {'label': 'measured'})
        plot_coords(ax[1], cam_im_coords[:, :, :], {'label': 'sim'})

        ax[1].set_ylabel('y (pix)')
        ax[1].set_xlabel('x (pix)')
        ax[1].legend(loc=1)
        ax[1].set_title(f'Image Cam | iter: {t}')

        plt.draw()
        plt.pause(1e-3)

        plt.figure(fig_tpm.number)
        ax_tpm.clear()
        tpm.plot(ax_tpm, fraction=1)
        # tpm.sample.plot(ax_tpm)
        viewplane = default_viewplane()
        x_sample, y_sample = viewplane.transform_points(tpm.sample.slide_top_plane.position_m)
        ax_tpm.set_xlim((x_sample - 3 * tpm.sample.outer_radius_m).detach(), (x_sample + 1.5*tpm.sample.slide_thickness_m).detach())
        ax_tpm.set_ylim((y_sample - 2 * tpm.sample.outer_radius_m).detach(), (y_sample + 2 * tpm.sample.outer_radius_m).detach())
        ax_tpm.set_aspect(1)

        plt.draw()
        plt.pause(1e-3)

for groupname in params:
    print('\n' + groupname + ':')
    for paramname in params[groupname]:
        if groupname == 'angle':
            print(f'  {paramname}: {params[groupname][paramname].detach().item():.3f}rad')
        else:
            print(f'  {paramname}: {format_prefix(params[groupname][paramname], ".3f")}m')


if do_plot_tube and iterations > 0:
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
    for groupname in params:
        for paramname in params_obj1_zshift_logs[groupname]:
            ax2.plot(params_obj1_zshift_logs[groupname][paramname], label=paramname)
    ax2.set_ylabel('Parameter (m | rad)')
    ax2.legend(loc=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Learning parameters')
    plt.show()
