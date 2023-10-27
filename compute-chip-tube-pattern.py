'''
Compute lumen-based organ-on-a-chip pattern.

This script computes phase correction patterns for several different locations inside a lumen-based
organ-on-a-chip sample, and saves those patterns as mat files. It optionally removes almost all
defocus and tilt. Not perfectly all, due to a slight bias from the way in which ray information is
sampled.
'''

import torch
from torch import tensor
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from plot_functions import format_prefix
from vector_functions import cartesian3d
from interpolate_shader import interpolate_shader
from ray_plane import CoordPlane
from testing import weighted_mean
from tpm import TPM
from sample_lumen_chip import SampleLumenChip
from dirconfig_raylearn import dirs


# Compute and save a phase correction pattern for specified location and wavelength
def compute_glass_tube_pattern(
        desired_focus_location_m, focus_location_label, remove_tilt,
        minimize_defocus_iterations, do_plot_backtrace, do_plot_phase_pattern):
    '''
    Compute a phase correction pattern for a glass tube

    Input
    -----
        desired_focus_location_m
            Vector. Desired focus location in tube with respect to tube center in meters.
        focus_location_label
            String. Label to indicate the focus location. Used for filename.
        remove_tilt
            Boolean. Remove mean tilt from pattern if True.
        remove_defocus_iterations
            Integer. How many iterations to minimize defocus. Skip defocus minimization if 0.
        do_plot_backtrace
            Boolean. Plot backtrace rays if True.
        do_plot_phase_pattern
            Boolean. Plot final phase pattern if True.
    '''

    # === Initialization === #
    # Initialize global coordinate system
    origin, x, y, z = cartesian3d()

    # Initilize TPM
    tpm = TPM()

    wavelength_m = 950e-9                       # Wavelength to compute phase from pathlength

    # Initialize standard tube properties
    # (Note: parameter scans may override these)
    tpm.sample = SampleLumenChip()
    tpm.sample.wavelength_m = wavelength_m
    tpm.sample.tube_yaw = 0.0                   # Rotate tube 90 deg
    #### Note! When the shader interpolator is fixed and doesn't transpose anymore, recheck

    # NA radius on the SLM, used for defocus optimization and tilt removal
    NA_radius_slm = (tpm.f5 / tpm.f7) * tpm.fobj1 * tpm.obj1_NA / tpm.slm_height_m

    # === Prepare defocus optimization === #
    tpm.backtrace_Nx = 100                      # Number of rays for defocus optimalization
    tpm.backtrace_Ny = 100

    # Initial sample z-shift, will be optimized for defocus
    tpm.sample_zshift = tensor((0.0,), requires_grad=True)

    # Define optimizer
    optimizer = torch.optim.Adam([
            {'lr': 2.0e-5, 'params': [tpm.sample_zshift]},
        ], lr=1.0e-5)

    errors = torch.zeros(minimize_defocus_iterations)

    # Iterable for optimization loop
    trange = tqdm(range(minimize_defocus_iterations), desc='error: -')

    for t in trange:
        # Update all dependent values
        tpm.update()
        tpm.sample.desired_focus_plane = CoordPlane(
            tpm.sample_plane.position_m + desired_focus_location_m,
            -x, y)

        # Raytrace back to SLM
        backrays_at_slm = tpm.backtrace()

        # Direction and NA mask of rays on SLM
        direction_xy_at_slm = tpm.slm_plane.transform_direction(backrays_at_slm.direction)
        NA_mask_for_obj_optim = compute_NA_mask(tpm.slm_plane, backrays_at_slm, NA_radius_slm)

        # Compute error from xy-direction
        error = direction_xy_at_slm[NA_mask_for_obj_optim.expand_as(direction_xy_at_slm)].std()
        error_value = error.detach().item()
        errors[t] = error_value

        trange.desc = f'error: {error_value:<8.3g}' \
            + f'sample z-shift: {format_prefix(tpm.sample_zshift, "8.3f")}m'

        # Gradient descent step
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Plot rays
        if t % 10 == 0 and t < 70 and do_plot_backtrace:
            plt.gca().clear()
            tpm.rays = tpm.backrays
            viewplane = CoordPlane(origin, z, x)
            tpm.plot(plt.gca(), viewplane=viewplane, fraction=0.005)
            plt.draw()
            plt.pause(1e-3)

    # === Prepare pattern interpolation === #
    tpm.update()
    tpm.sample.desired_focus_plane = CoordPlane(
        tpm.sample_plane.position_m + desired_focus_location_m,
        -x, y)
    tpm.backtrace_Nx = 500                      # Number of rays for pattern interpolation
    tpm.backtrace_Ny = 500
    backrays_at_slm = tpm.backtrace()

    # Remove tilt
    if remove_tilt:
        # Compute mask of ray coords within NA circle
        NA_mask = compute_NA_mask(tpm.slm_plane, backrays_at_slm, NA_radius_slm)
        mean_direction = weighted_mean(backrays_at_slm.direction, NA_mask.expand_as(backrays_at_slm.direction), dim=(-3, -2))

        # Compute mean tilt pattern, with z as optical axis, based on mean ray direction within NA
        tilt_m = (
                backrays_at_slm.position_m[:, :, 0] * mean_direction[0] / mean_direction[2]
                + backrays_at_slm.position_m[:, :, 1] * mean_direction[1] / mean_direction[2]
            ).unsqueeze(-1)

    else:
        tilt_m = 0

    # Compute screen coordinates and pathlength with mean tilt subtracted
    coords = tpm.slm_plane.transform_rays(backrays_at_slm).detach()
    pathlength_to_slm_at_coords = backrays_at_slm.pathlength_m.detach() + tilt_m

    # Compute field with interpolate_shader
    data = torch.cat((coords, pathlength_to_slm_at_coords), 2)
    slm_edge = 0.5                                      # Center to top of pattern, in slm heights
    extent = (-slm_edge, slm_edge, -slm_edge, slm_edge)
    field_SLM, phase_SLM = interpolate_shader(
        data.detach().numpy(),
        npoints=(tpm.slm_height_pixels, tpm.slm_height_pixels),
        limits=extent,
        wavelength_m=wavelength_m,
        )

    NA_radius_slm = (tpm.f5 / tpm.f7) * tpm.fobj1 * tpm.obj1_NA / tpm.slm_height_m

    # Plot
    if do_plot_phase_pattern:
        # Plot SLM phase correction pattern
        fig_phase_pattern = plt.figure(dpi=200, figsize=(6, 5.5))
        plt.imshow(np.angle(field_SLM), extent=extent, vmin=-np.pi, vmax=np.pi,
                   cmap='twilight', interpolation='nearest')
        plt.title(f'Phase pattern for {focus_location_label}, λ={format_prefix(wavelength_m)}m')

        # Draw a circle to indicate NA
        N_verts_NA_circle = 200
        theta_vert_NA_circle = np.linspace(0, 2*np.pi, N_verts_NA_circle)
        x_NA_circle = NA_radius_slm * np.cos(theta_vert_NA_circle)
        y_NA_circle = NA_radius_slm * np.sin(theta_vert_NA_circle)
        plt.plot(x_NA_circle, y_NA_circle, '--g', linewidth=2.5, label=f'NA={tpm.obj1_NA}')
        plt.legend(loc='upper right')

        plt.xticks(())
        plt.yticks(())

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)      # Colobar Axes
        colorbar = plt.colorbar(ticks=[-np.pi, 0, np.pi], label='Phase', cax=cax)
        colorbar.ax.set_yticklabels(['−π', '0', '+π'])
        plt.show()

    # Compute NA mask
    Mx = tpm.slm_height_pixels
    My = tpm.slm_height_pixels
    x_lin_SLM = torch.linspace(-0.5, 0.5, Mx).view(Mx, 1) * (Mx != 1)
    y_lin_SLM = torch.linspace(-0.5, 0.5, My).view(1, My) * (My != 1)
    NA_mask_SLM = NA_radius_slm > (x_lin_SLM * x_lin_SLM + y_lin_SLM * y_lin_SLM).sqrt()

    # Compute RMS wavefront error
    phase_SLM_t = tensor(phase_SLM)
    phase_SLM_t_m = phase_SLM_t - phase_SLM_t.mean()        # Subtract mean
    RMS_wavefront_error_rad = ((phase_SLM_t_m * NA_mask_SLM).pow(2).sum() / NA_mask_SLM.sum()).sqrt()
    print(f'RMS wavefront error: {RMS_wavefront_error_rad:.2f} rad')

    # Save file
    matpath_out = \
        str(dirs['localdata'].joinpath('raylearn-data/TPM/slm-patterns/pattern-chip-'
            + f'{focus_location_label}-λ{format_prefix(wavelength_m, formatspec=".0f")}m.mat'))
    mdict = {
        'focus_location_label': focus_location_label,
        'field_SLM': field_SLM,
        'phase_SLM': phase_SLM,
        'obj1_zshift': tpm.obj1_zshift.detach().numpy(),
        'sample_zshift': tpm.sample_zshift.detach().numpy(),
        'wavelength_m': wavelength_m,
        'obj1_NA': tpm.obj1_NA,
        'slm_height_m': tpm.slm_height_m,
        'x_lin_SLM': x_lin_SLM.detach().numpy(),
        'y_lin_SLM': y_lin_SLM.detach().numpy(),
        'NA_mask_SLM': NA_mask_SLM.detach().numpy(),
        'RMS_wavefront_error_rad': RMS_wavefront_error_rad.detach().numpy(),
        'desired_focus_location_m': desired_focus_location_m.numpy(),
    }

    matfile_out = hdf5storage.savemat(matpath_out, mdict)
    print(f'Saved to {matpath_out}')


def compute_NA_mask(plane, rays, NA_radius):
    '''Compute boolean mask for positions inside NA circle.'''
    position = plane.transform_points(rays.position_m)
    radius_at_slm = position.pow(2).sum(dim=-1, keepdim=True).sqrt()
    return NA_radius > radius_at_slm


# === Settings === #
# General settings
do_plot_backtrace = False
do_plot_phase_pattern = True

# Desired focus location relative to exact tube center (in meters)

compute_glass_tube_pattern(
    desired_focus_location_m=tensor((0., 0., 180e-6,)),
    focus_location_label='bottom',
    remove_tilt=True,
    minimize_defocus_iterations=200,
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)

compute_glass_tube_pattern(
    desired_focus_location_m=tensor((0., 0., 0.,)),
    focus_location_label='center',
    remove_tilt=True,
    minimize_defocus_iterations=200,
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)

compute_glass_tube_pattern(
    desired_focus_location_m=tensor((0., 0., 0.,)),
    focus_location_label='center-with-defocus',
    remove_tilt=True,
    minimize_defocus_iterations=0,
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)

compute_glass_tube_pattern(
    desired_focus_location_m=tensor((0., 0., -180e-6,)),
    focus_location_label='top',
    remove_tilt=True,
    minimize_defocus_iterations=200,
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)

compute_glass_tube_pattern(
    desired_focus_location_m=tensor((0., 180e-6, 0.,)),
    focus_location_label='side',
    remove_tilt=True,
    minimize_defocus_iterations=200,
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)

compute_glass_tube_pattern(
    desired_focus_location_m=tensor((0., 180e-6, 0.,)),
    focus_location_label='side-with-tilt',
    remove_tilt=False,
    minimize_defocus_iterations=200,
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)
