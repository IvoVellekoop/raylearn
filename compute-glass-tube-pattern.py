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
from ray_plane import CoordPlane, translate
from testing import MSE
from tpm import TPM
from sample_tube import SampleTube
from dirconfig_raylearn import dirs


# Compute and save a phase correction pattern for specified location and wavelength
def compute_glass_tube_pattern(desired_focus_location_from_tube_center_m, focus_location_label,
                               do_plot_backtrace, do_plot_phase_pattern):

    # === Initialization === #
    # Initialize global coordinate system
    origin, x, y, z = cartesian3d()

    # Initilize TPM
    tpm = TPM()
    tpm.update()

    wavelength_m = 808e-9                       # Wavelength to compute phase from pathlength

    # Initialize standard tube properties
    # (Note: parameter scans may override these)
    tpm.sample = SampleTube()
    tpm.sample.tube_angle = np.pi / 2           # Rotate tube 90 deg
    #### Note! When the shader interpolator is fixed and doesn't transpose anymore, recheck
    tpm.sample.n_tube = 1.5106                  # Schott N-BK7 @808nm
    tpm.sample.n_slide = 1.5170                 # Soda Lime Glass @808nm
    tpm.sample.n_inside = 1.3290                # Water inside the tube @808nm, 25degC
    tpm.sample.n_outside = 1.3290               # Water between tube and slide @808nm, 25degC


    # === Prepare defocus optimization === #
    tpm.backtrace_Nx = 100                      # Number of rays for defocus optimalization
    tpm.backtrace_Ny = 100

    # Initial sample z-shift, will be optimized for defocus
    tpm.sample_zshift = tensor((0.0,), requires_grad=True)
    tpm.update()

    # Define optimizer
    optimizer = torch.optim.Adam([
            {'lr': 2.0e-5, 'params': [tpm.sample_zshift]},
        ], lr=1.0e-5)

    iterations = 200
    errors = torch.zeros(iterations)

    # Iterable for optimization loop
    trange = tqdm(range(iterations), desc='error: -')

    for t in trange:
        # Update all dependent values
        tpm.update()

        # Place focus plane at center of tube + translated with desired vector
        tpm.sample.desired_focus_plane = \
            translate(tpm.sample_plane,
                    desired_focus_location_from_tube_center_m
                    - tpm.sample_plane.normal * tpm.sample.outer_radius_m)

        tpm.sample.desired_focus_plane = CoordPlane(
            tpm.sample.cyl_plane.position_m + desired_focus_location_from_tube_center_m,
            -x, y)    # Opening tan(angle)=1

        # Raytrace back to SLM
        backrays_at_slm = tpm.backtrace()

        # Extract position and directio of rays
        position_xy_at_slm = tpm.slm_plane.transform_points(backrays_at_slm.position_m)
        direction_xy_at_slm = tpm.slm_plane.transform_direction(backrays_at_slm.direction)
        direction_xy_at_slm.retain_grad()

        # Compute mask of ray coords within NA circle
        radius_at_slm = position_xy_at_slm.pow(2).sum(dim=-1, keepdim=True).sqrt()
        NA_radius_slm = (tpm.f5 / tpm.f7) * tpm.fobj1 * tpm.obj1_NA / tpm.slm_height_m
        NA_mask_for_obj_optim = NA_radius_slm > radius_at_slm

        # Compute error from xy-direction
        error = MSE(direction_xy_at_slm * NA_mask_for_obj_optim, 0)
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
            tpm.plot(plt.gca(), fraction=0.01)
            plt.draw()
            plt.pause(1e-3)

    # === Prepare pattern interpolation === #
    tpm.backtrace_Nx = 500                      # Number of rays for pattern interpolation
    tpm.backtrace_Ny = 500
    # tpm.update()          #### Why does this mess it up?
    backrays_at_slm = tpm.backtrace()

    # Propagate to screen and compute screen coordinates
    coords = tpm.slm_plane.transform_rays(backrays_at_slm).detach()

    # Compute field with interpolate_shader
    pathlength_to_slm_at_coords = backrays_at_slm.pathlength_m.detach()
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
        plt.title('Phase pattern, ' + f'λ={format_prefix(wavelength_m)}m')

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
    matpath_out = str(dirs['localdata'].joinpath('raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-'
        + f'{focus_location_label}'
        + f'-λ{format_prefix(wavelength_m, formatspec=".0f")}m.mat'))
    mdict = {
        'focus_location_label': focus_location_label,
        'field_SLM': field_SLM,
        'phase_SLM': phase_SLM,
        'obj1_zshift': tpm.obj1_zshift.detach().numpy(),
        'wavelength_m': wavelength_m,
        'obj1_NA': tpm.obj1_NA,
        'slm_height_m': tpm.slm_height_m,
        'x_lin_SLM': x_lin_SLM.detach().numpy(),
        'y_lin_SLM': y_lin_SLM.detach().numpy(),
        'NA_mask_SLM': NA_mask_SLM.detach().numpy(),
        'RMS_wavefront_error_rad': RMS_wavefront_error_rad.detach().numpy(),
        'desired_focus_location_from_tube_center_m': desired_focus_location_from_tube_center_m.numpy(),
    }

    matfile_out = hdf5storage.savemat(matpath_out, mdict)
    print(f'Saved to {matpath_out}')


# === Settings === #
# General settings
do_plot_backtrace = False
do_plot_phase_pattern = False

# Desired focus location relative to exact tube center (in meters)

compute_glass_tube_pattern(
    desired_focus_location_from_tube_center_m=tensor((0., 0., 51e-6,)),
    focus_location_label='bottom',
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)

compute_glass_tube_pattern(
    desired_focus_location_from_tube_center_m=tensor((0., 0., 0.,)),
    focus_location_label='center',
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)

compute_glass_tube_pattern(
    desired_focus_location_from_tube_center_m=tensor((0., 0., -51e-6,)),
    focus_location_label='top',
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)

compute_glass_tube_pattern(
    desired_focus_location_from_tube_center_m=tensor((0., 53.25e-6, 0.,)),
    focus_location_label='side-with-tilt',
    do_plot_backtrace=do_plot_backtrace,
    do_plot_phase_pattern=do_plot_phase_pattern)
