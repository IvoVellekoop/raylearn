import torch
from torch import tensor
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plot_functions import format_prefix
from vector_functions import cartesian3d
from ray_plane import translate
from interpolate_shader import interpolate_shader
from tpm import TPM
from dirconfig_raylearn import dirs

# General settings
do_plot_phase_pattern = True

# Focus settings
wavelength_m = 808e-9
n_water = 1.3290                # Water inside the tube @808nm, 25degC
x_focus_shift_m = 0e-6
y_focus_shift_m = 0e-6
z_focus_shift_m = 30e-6

# Initialize global coordinate system
origin, x, y, z = cartesian3d()

# Initilize TPM
tpm = TPM()
tpm.update()
tpm.sample.n_inside = n_water   # Refractive index of focus medium
translate_focus_vector_m = x * x_focus_shift_m + y * y_focus_shift_m + z * z_focus_shift_m
tpm.sample.desired_focus_plane = translate(tpm.sample_plane, translate_focus_vector_m)

tpm.backtrace_Nx = 500
tpm.backtrace_Ny = 500
tpm.backtrace_source_opening_tan_angle = 1.2 * np.tan(np.arcsin(tpm.obj1_NA / tpm.n_water))
tpm.update()
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
matpath_out = str(dirs['localdata'].joinpath('raylearn-data/TPM/slm-patterns/pattern-focusshift_'
    + f'x{x_focus_shift_m * 1e6:.0f}um_'
    + f'y{y_focus_shift_m * 1e6:.0f}um_'
    + f'z{z_focus_shift_m * 1e6:.0f}um_'
    + f'λ{format_prefix(wavelength_m, formatspec=".0f")}m.mat'))
mdict = {
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
    'translate_focus_vector_m': translate_focus_vector_m.numpy(),
}

matfile_out = hdf5storage.savemat(matpath_out, mdict)
print(f'Saved to {matpath_out}')
