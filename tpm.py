"""
The Two Photon Microscope.
"""

import torch
from torch import tensor
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torchviz import make_dot

from vector_functions import rotate, cartesian3d
from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays, plot_coords, format_prefix
from optical import ideal_lens, snells, galvo_mirror, slm_segment
from testing import MSE


# Set default tensor type to double (64 bit)
# Machine epsilon of float (32-bit) is 2^-23 = 1.19e-7
# The ray simulation contains both meter and micrometer scales,
# hence floats might not be precise enough.
# https://en.wikipedia.org/wiki/Machine_epsilon
torch.set_default_tensor_type('torch.DoubleTensor')

plt.rc('font', size=12)


class TPM(): #torch.nn.Module):
    """
    The Two Photon Microscope.

    Conveniently pack all variables that make up the TPM in one class.
    """

    def __init__(self):
        """
        Define all properties of all optical elements as initial guess.

        All statically defined properties should be defined here.
        Properties that depend on other properties should be computed in
        the update method. All lengths in meters.

        The measurements define the SLM segment and Galvo Mirror settings,
        and are hence included.

        Note: For now, the TPM alignment up to the SLM is assumed to be
        perfect. The magnification between Galvo Mirrors and the SLM is 1.
        Hence the Galvo Mirrors and SLM are all defined on the same position,
        skipping the 4F system lenses in between them.
        """
        #super().__init__()

        # Define refractive indices at wavelength of 715nm
        # References:
        # https://refractiveindex.info/?shelf=main&book=H2O&page=Hale
        # https://refractiveindex.info/?shelf=glass&book=soda-lime&page=Rubin-clear
        # https://refractiveindex.info/?shelf=glass&book=SCHOTT-multipurpose&page=D263TECO
        self.n_water = 1.3304
        self.n_coverslip = 1.5185

        # Define coordinate system
        self.coordsystem = cartesian3d()
        origin, x, y, z = self.coordsystem

        # Galvo Mirrors
        # Thorlabs GVS111(/M)
        # https://bmpi.wiki.utwente.nl/doku.php?id=instrumentation:galvo:galvo_scanners
        # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=7616&pn=GVS111/M#7617
        # Optical scan angle = 2x mechanical angle
        # https://docs.scanimage.org/Configuration/Scanners/Resonant%2BScanner.html
        # https://bmpi.wiki.utwente.nl/lib/exe/fetch.php?media=instrumentation:galvo:gvs111_m-manual.pdf
        galvo_volts_per_optical_degree = 0.5
        ########### galvo_volts_per_mechanical_degree = #############
        self.galvo_rad_per_V = (np.pi/180) / galvo_volts_per_optical_degree
        self.galvo_angle = tensor((0.067,))            # Rotation angle around optical axis

        # SLM
        # Meadowlark 1920x1152 XY Phase Series
        # https://bmpi.wiki.utwente.nl/doku.php?id=instrumentation:slm:meadowlark_slm
        # https://www.meadowlark.com/store/data_sheet/SLM%20-%201920%20x%201152%20Data%20Sheet%20021021.pdf
        # https://www.meadowlark.com/images/files/Specification%20Backgrounder%20for%20XY%20Series%20Phase%20SLMS%20-%20SB0520.pdf
        self.slm_width = 10.7e-3
        self.slm_height = 10.7e-3
        self.slm_angle = tensor((-0.030,))              # Rotation angle around optical axis
        self.slm_zshift = tensor((0.,))

        # Coverslip
        self.coverslip_thickness = tensor((170e-6,))

        # Focal distances (m)
        self.f5 = 150e-3
        self.f7 = 300e-3
        self.f9 = 150e-3
        self.f10 = 200e-3
        self.f11 = 100e-3
        self.obj1_tubelength = 200e-3           # Objective standard tubelength
        self.obj1_magnification = 16            # Objective magnification
        self.fobj1 = self.obj1_tubelength / self.obj1_magnification
        # self.fobj1 = 13.8e-3  #### Measured
        self.obj2_tubelength = 165e-3           # Objective standard tubelength
        self.obj2_magnification = 100           # Objective magnification
        # self.fobj2 = self.obj2_tubelength / self.obj2_magnification
        self.fobj2 = 1.621e-3

        # Lens planes transmission arm
        self.sample_zshift = tensor((0.,))
        self.obj2_zshift = tensor((0.,))
        self.L9_zshift = tensor((0.,))
        self.L10_zshift = tensor((0.,))

        # Camera planes
        # Basler acA2000-165umNIR
        # https://www.baslerweb.com/en/products/cameras/area-scan-cameras/ace/aca2000-165umnir/
        self.cam_pixel_size = 5.5e-6
        self.cam_ft_xshift = tensor((2.7e-3,))
        self.cam_ft_yshift = tensor((2.83e-3,))
        self.cam_im_xshift = tensor((0.,))
        self.cam_im_yshift = tensor((0.,))
        self.cam_im_zshift = tensor((0.,))

    def set_measurement(self, matfile):
        # SLM coords and Galvo rotations
        self.slm_coords = tensor(matfile['p/rects'])[0:2, :].T.view(-1, 2)
        self.galvo_volts = tensor((matfile['p/galvoXs'], matfile['p/galvoYs'])).T \
                          - tensor((matfile['p/GalvoXcenter'], matfile['p/GalvoYcenter'])).view(1, 1, 2)
        ######### Correct with SLM ppp instead
        self.galvo_rots = self.galvo_volts * self.galvo_rad_per_V

    def update(self):
        """
        Update dependent properties. These properties are defined as a function
        of statically defined properties and/or other dependent properties.

        Properties that depend on dynamic properties should be computed here,
        so they get recomputed whenever update is called. All lengths in meters.
        """
        origin, x, y, z = self.coordsystem

        # SLM
        self.slm_x = rotate(x * self.slm_width, z, self.slm_angle)
        self.slm_y = rotate(y * self.slm_height, z, self.slm_angle)
        self.slm_plane = CoordPlane(origin + self.slm_zshift * z, self.slm_x, self.slm_y)

        # Galvo
        self.galvo_x = rotate(x, z, self.galvo_angle)
        self.galvo_y = rotate(y, z, self.galvo_angle)
        self.galvo_plane = CoordPlane(origin, self.galvo_x, self.galvo_y)

        # Lens planes to sample plane
        self.L5 = Plane(origin + self.f5*z, -z)
        self.L7 = Plane(self.L5.position_m + (self.f5 + self.f7)*z, -z)
        self.OBJ1 = Plane(self.L7.position_m + (self.f7 + self.fobj1)*z, -z)

        # Sample plane and coverslip
        coverslip_front_to_sample_plane = (170e-6 - self.coverslip_thickness) * z
        self.coverslip_front_plane = CoordPlane(
            self.OBJ1.position_m + self.fobj1*z + coverslip_front_to_sample_plane, -x, y)
        self.sample_plane = CoordPlane(self.OBJ1.position_m + self.fobj1*z +
                                       self.sample_zshift * z, -x, y)

        # Objective
        self.OBJ2 = Plane(self.sample_plane.position_m + self.fobj2*z + self.obj2_zshift*z, -z)
        self.L9 = Plane(self.OBJ2.position_m + (self.fobj2 + self.f9 + self.L9_zshift)*z, -z)
        self.L10 = Plane(self.L9.position_m + (self.f9 + self.f10)*z + self.L10_zshift*z, -z)

        # Cameras
        self.cam_im_plane = CoordPlane(self.L9.position_m + self.f9*z +
                                       self.cam_im_xshift*x + self.cam_im_yshift*y +
                                       self.cam_im_zshift*z,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * y)
        self.cam_ft_plane = CoordPlane(self.L10.position_m + self.f10*z +
                                       self.cam_ft_xshift*x + self.cam_ft_yshift*y,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * -y)

    def raytrace(self):
        """
        Forward ray tracing based on current properties.

        Output
        ------
            cam_ft_coords   SX x SY x GX x GY x 2 Vector, where SX/SY and GX/GY
                            denote the number of SLM segments and Galvo angles
                            for that corresponding dimension. Predicted camera
                            coordinates of Fourier plane camera.
            cam_im_coords   SX x SY x GX x GY x 2 Vector, where SX/SY and GX/GY
                            denote the number of SLM segments and Galvo angles
                            for that corresponding dimension.  Predicted camera
                            coordinates of Image plane camera.
        """
        # Initial Ray list
        origin, x, y, z = self.coordsystem
        self.rays = [Ray(origin, -z)]

        # Propagation to objective 1
        self.rays.append(galvo_mirror(self.rays[-1], self.galvo_plane, self.galvo_rots))
        self.rays.append(self.rays[-1].intersect_plane(self.slm_plane))
        self.rays.append(slm_segment(self.rays[-1], self.slm_plane, self.slm_coords))
        self.rays.append(ideal_lens(self.rays[-1], self.L5, self.f5))
        self.rays.append(ideal_lens(self.rays[-1], self.L7, self.f7))
        self.rays.append(ideal_lens(self.rays[-1], self.OBJ1, self.fobj1))
        self.rays.append(self.rays[-1].copy(refractive_index=self.n_water))

        # Propagation through coverslip
        self.rays.append(self.rays[-1].intersect_plane(self.coverslip_front_plane))
        self.rays.append(snells(self.rays[-1], self.coverslip_front_plane.normal, self.n_coverslip))
        self.rays.append(self.rays[-1].intersect_plane(self.sample_plane))
        # self.rays.append(snells(self.rays[-1], self.sample_plane.normal, 1.))

        # # Propagation from objective 2
        self.rays.append(self.rays[-1].copy(refractive_index=1.0))
        self.rays.append(ideal_lens(self.rays[-1], self.OBJ2, self.fobj2))
        self.rays.append(ideal_lens(self.rays[-1], self.L9, self.f9))

        # Propagation onto cameras
        cam_im_ray = self.rays[-1].intersect_plane(self.cam_im_plane)
        self.rays.append(cam_im_ray)
        self.rays.append(ideal_lens(self.rays[-1], self.L10, self.f10))
        cam_ft_ray = self.rays[-1].intersect_plane(self.cam_ft_plane)
        self.rays.append(cam_ft_ray)

        # Cameras
        self.cam_ft_coords = self.cam_ft_plane.transform_rays(cam_ft_ray)
        self.cam_im_coords = self.cam_im_plane.transform_rays(cam_im_ray)

        return self.cam_ft_coords, self.cam_im_coords

    def plot(self, ax=plt.gca()):
        """Plot the TPM setup and the current rays."""

        # Plot lenses and planes
        scale = 0.008
        plot_plane(ax, self.slm_plane, 0.8, ' SLM', plotkwargs={'color': 'red'})
        plot_plane(ax, self.galvo_plane, scale, ' Galvo', plotkwargs={'color': 'red'})
        plot_lens(ax, self.L5, self.f5, scale, ' L5\n')
        plot_lens(ax, self.L7, self.f7, scale, ' L7\n')

        plot_lens(ax, self.OBJ1, self.fobj1, scale, ' OBJ1\n')
        plot_plane(ax, self.coverslip_front_plane, scale*0.8, '', ' coverslip\n front')
        plot_plane(ax, self.sample_plane, scale*0.7, '', ' sample plane')
        plot_lens(ax, self.OBJ2, self.fobj2, 0.75*scale, 'OBJ2\n')

        plot_lens(ax, self.L9, self.f9, scale, ' L9\n')
        plot_lens(ax, self.L10, self.f10, scale, 'L10\n')

        plot_plane(ax, self.cam_ft_plane, 1000, ' Fourier Cam', plotkwargs={'color': 'red'})
        plot_plane(ax, self.cam_im_plane, 1000, ' Image Cam', plotkwargs={'color': 'red'})

        # Plot rays
        # ray2_exp_pos = self.rays[2].position_m.expand(self.rays[3].position_m.shape)
        # ray2_exp = self.rays[2].copy(position_m=ray2_exp_pos)
        # raylist = [ray2_exp] + self.rays[3:]
        plot_rays(ax, self.rays, fraction=0.03)


# Import measurement
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-400um/raylearn_pencil_beam_738477.786123_400um.mat'
matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-empty/raylearn_pencil_beam_738477.729080_empty.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/10-Feb-2022-empty/raylearn_pencil_beam_738562.645439_empty.mat'
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
# tpm.slm_angle = tensor((0.,), requires_grad=True)
# tpm.galvo_angle = tensor((0.,), requires_grad=True)
tpm.cam_ft_xshift = tensor((0.,), requires_grad=True)
tpm.cam_ft_yshift = tensor((0.,), requires_grad=True)
tpm.cam_im_xshift = tensor((2.74e-3,), requires_grad=True)
tpm.cam_im_yshift = tensor((-2.68e-3,), requires_grad=True)
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
        {'lr': 1.0e-3, 'params': params['angle'].values()},
        {'lr': 2.0e-4, 'params': params['objective'].values()},
        {'lr': 2.0e-3, 'params': params['other'].values()},
    ], lr=1.0e-5)

iterations = 500
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
        + 1e-1 * MSE(cam_im_coords_gt, cam_im_coords) \

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
matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-400um/raylearn_pencil_beam_738477.768870_400um.mat'
# matpath = 'LocalData/raylearn-data/TPM/pencil-beam-positions/17-Nov-2021-empty/raylearn_pencil_beam_738477.729080_empty.mat'
matfile = h5py.File(matpath, 'r')

cam_ft_coords_gt = tensor((matfile['cam_ft_col'], matfile['cam_ft_row'])).permute(1, 2, 0)
cam_im_coords_gt = tensor((matfile['cam_img_col'], matfile['cam_img_row'])).permute(1, 2, 0)

# Parameters
tpm.coverslip_thickness = tensor((300e-6,), requires_grad=True)

params_coverslip = {
    'Coverslip Thickness': tpm.coverslip_thickness,
}

tpm.set_measurement(matfile)
tpm.update()

# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 1.0e-4, 'params': params_coverslip.values()},
    ], lr=1.0e-5)

iterations = 250
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
    if t % 40 == 0 and True:
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


# tpm.plot()
# 
# 
# fig, ax1 = plt.subplots(figsize=(7, 7))
# fig.dpi = 144
# 
# 
# # Plot error
# errorcolor = 'tab:red'
# RMSEs = np.sqrt(errors.detach().cpu())
# ax1.plot(RMSEs, label='error', color=errorcolor)
# ax1.set_ylabel('Error (pix)')
# ax1.set_ylim((0, max(RMSEs)))
# ax1.legend()

# ax2 = ax1.twinx()
# for name in params_coverslip_log:
#     ax2.plot(params_coverslip_log[name]*1e6, label=name)
# ax2.set_ylabel('Parameter (um | rad)')
# ax2.legend()

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title('Learning parameters')
# plt.show()


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
