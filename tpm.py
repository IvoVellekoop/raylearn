"""
The Two Photon Microscope with a Grid Target.

Fit measurements of an absorbing Grid Target in the Two Photon Microscope.
"""

import torch
from torch import tensor, stack, meshgrid
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
# from torchviz import make_dot

import vector_functions
from vector_functions import norm, rotate
from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays, plot_coords
from optical import ideal_lens, snells, galvo_mirror, slm_segment, intensity_mask_smooth_grid
from testing import MSE


# Set default tensor type to double (64 bit)
# Machine epsilon of float (32-bit) is 2^-23 = 1.19e-7
# The ray simulation contains both meter and micrometer scales,
# hence floats might not be precise enough.
# https://en.wikipedia.org/wiki/Machine_epsilon
torch.set_default_tensor_type('torch.DoubleTensor')

plt.rc('font', size=12)


class TPM(torch.nn.Module):
    """
    The Two Photon Microscope.

    Conveniently pack all variables that make up the TPM in one class.
    """

    def __init__(self):
        """
        Define all properties of all optical elements as initial guess.

        All properties that remain untouched or will be directly overwritten
        should be defined here. Properties that depend on other properties
        should be computed in the update method. All lengths in meters.

        The measurements define the SLM segment and Galvo Mirror settings,
        and are hence included.

        Note: For now, the TPM alignment up to the SLM is assumed to be
        perfect. The magnification between Galvo Mirrors and the SLM is 1.
        Hence the Galvo Mirrors and SLM are all defined on the same position,
        skipping the 4F system lenses in between them.
        """
        super().__init__()

        # Define refractive indices at wavelength of 715nm
        # References:
        # https://refractiveindex.info/?shelf=main&book=H2O&page=Hale
        # https://refractiveindex.info/?shelf=glass&book=soda-lime&page=Rubin-clear
        # https://refractiveindex.info/?shelf=glass&book=SCHOTT-multipurpose&page=D263TECO
        self.n_water = 1.3304
        self.n_target = 1.5191
        self.n_coverslip = 1.5185

        # Define coordinate system
        origin = tensor((0., 0., 0.))
        x = tensor((1., 0., 0.))
        y = tensor((0., 1., 0.))
        z = tensor((0., 0., 1.))

        self.coordsystem = (origin, x, y, z)

        # Galvo
        self.galvo_rad_per_V = (np.pi/180) / 0.5
        self.galvo_angle = tensor((0.,))            # Rotation angle around optical axis

        # SLM
        self.slm_width = 10.7e-3
        self.slm_height = 10.7e-3
        self.slm_angle = tensor((0.,))              # Rotation angle around optical axis

        # SLM coords and Galvo rotations
        self.slm_coords = tensor(matfile['p/rects'])[0:2, :].T.view(-1, 2)
        self.galvo_volts = tensor((matfile['p/galvoXs'], matfile['p/galvoYs'])).T \
                          - tensor((matfile['p/GalvoXcenter'], matfile['p/GalvoYcenter'])).view(1, 1, 2)
        self.galvo_rots = self.galvo_volts * self.galvo_rad_per_V

        # Focal distances (m)
        self.f5 = 150e-3
        self.f7 = 300e-3
        self.f9 = 150e-3
        self.f10 = 200e-3
        self.f11 = 100e-3
        self.obj1_tubelength = 200e-3           # Objective standard tubelength
        self.obj1_magnification = 16            # Objective magnification
        self.fobj1 = self.obj1_tubelength / self.obj1_magnification
        self.obj2_tubelength = 165e-3           # Objective standard tubelength
        self.obj2_magnification = 100           # Objective magnification
        self.fobj2 = self.obj2_tubelength / self.obj2_magnification

        # Lens planes to sample plane
        self.L5 = Plane(origin + self.f5*z, -z)
        self.L7 = Plane(self.L5.position_m + (self.f5 + self.f7)*z, -z)
        self.OBJ1 = Plane(self.L7.position_m + (self.f7 + self.fobj1)*z, -z)

        # Grid Target
        # https://www.thorlabs.com/thorproduct.cfm?partnumber=R1L3S3P
        self.grid_target_x_spacing = 50e-6
        self.grid_target_y_spacing = 50e-6
        self.grid_target_thickness = 0.
        self.grid_target_zshift = tensor((0.,))

        # Lens planes transmission arm
        self.obj2_zshift = tensor((0.,))
        self.L9_zshift = tensor((0.,))

        # Camera planes
        self.cam_pixel_size = 5.5e-6
        self.cam_ft_shift = tensor((0., 0., 0.))
        self.cam_im_shift = tensor((0., 0., 0.))

    def update(self):
        """
        Update dependent properties. These depend on parameters to be learned.

        Properties that depend on other properties should be computed here,
        so they get recomputed whenever update is called. All lengths in meters.
        """
        origin, x, y, z = self.coordsystem

        # SLM
        self.slm_x = rotate(x * self.slm_width, z, self.slm_angle)
        self.slm_y = rotate(y * self.slm_height, z, self.slm_angle)
        self.slm_plane = CoordPlane(origin, self.slm_x, self.slm_y)

        # Galvo
        self.galvo_x = rotate(x, z, self.galvo_angle)
        self.galvo_y = rotate(y, z, self.galvo_angle)
        self.galvo_plane = CoordPlane(origin, self.galvo_x, self.galvo_y)
        
        # Grid target
        self.sample_plane = CoordPlane(self.OBJ1.position_m + self.fobj1*z, -x, y)
        self.grid_target_back_plane = CoordPlane(
            self.sample_plane.position_m + z*self.grid_target_zshift,
            self.grid_target_x_spacing * -x,
            self.grid_target_y_spacing * y)
        self.grid_target_front_plane = Plane(
            self.grid_target_back_plane.position_m - self.grid_target_thickness*z, -z)


        # Objective
        self.OBJ2 = Plane(self.sample_plane.position_m + self.fobj2*z + self.obj2_zshift*z, -z)
        self.L9 = Plane(self.OBJ2.position_m + (self.fobj2 + self.f9)*z + self.L9_zshift, -z)
        self.L10 = Plane(self.L9.position_m + (self.f9 + self.f10)*z, -z)
        self.L11 = Plane(self.L10.position_m + (self.f10 + self.f11)*z, -z)

        # Cameras
        self.cam_ft_plane = CoordPlane(self.L10.position_m + self.f10*z + self.cam_ft_shift,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * y)
        self.cam_im_plane = CoordPlane(self.L11.position_m + self.f11*z + self.cam_im_shift,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * y)


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
        self.rays = [Ray(origin, -self.z)]

        # Propagation to objective 1
        self.rays.append(galvo_mirror(self.rays[-1], self.galvo_plane, self.galvo_rots))
        self.rays.append(slm_segment(self.rays[-1], self.slm_plane, self.slm_coords))
        self.rays.append(ideal_lens(self.rays[-1], self.L5, self.f5))
        self.rays.append(ideal_lens(self.rays[-1], self.L7, self.f7))
        self.rays.append(ideal_lens(self.rays[-1], self.OBJ1, self.fobj1))
        self.rays.append(self.rays[-1].copy(refractive_index=self.n_water))

        # Propagation through grid target glass slide
        self.rays.append(self.rays[-1].intersect_plane(self.grid_target_front_plane))
        self.rays.append(snells(self.rays[-1], self.grid_target_front_plane.normal, self.n_target))
        self.rays.append(self.rays[-1].intersect_plane(self.grid_target_back_plane))
        self.rays.append(intensity_mask_smooth_grid(self.rays[-1], self.grid_target_back_plane, 4))

        # Propagation from objective 2
        self.rays.append(snells(self.rays[-1], self.grid_target_back_plane.normal, self.n_coverslip))
        self.rays.append(self.rays[-1].copy(refractive_index=1.0))
        self.rays.append(ideal_lens(self.rays[-1], self.OBJ2, self.fobj2))
        self.rays.append(ideal_lens(self.rays[-1], self.L9, self.f9))
        self.rays.append(ideal_lens(self.rays[-1], self.L10, self.f10))
        cam_ft_ray = self.rays[-1].intersect_plane(self.cam_ft_plane)
        self.rays.append(cam_ft_ray)
        self.rays.append(ideal_lens(self.rays[-1], self.L11, self.f11))
        cam_im_ray = self.rays[-1].intersect_plane(self.cam_im_plane)
        self.rays.append(cam_im_ray)

        # Cameras
        self.cam_ft_coords = self.cam_ft_plane.transform(cam_ft_ray)
        self.cam_im_coords = self.cam_im_plane.transform(cam_im_ray)

        return self.cam_ft_coords, self.cam_im_coords, cam_ft_ray.intensity, cam_im_ray.intensity

    def plot(self):
        """Plot the TPM setup and the current rays."""
        fig = plt.figure(figsize=(15, 4))
        fig.dpi = 144
        ax1 = plt.gca()

        # Plot lenses and planes
        scale = 0.020
        plot_plane(ax1, self.slm_plane, 1, ' SLM')
        plot_plane(ax1, self.galvo_plane, scale, ' GM')
        plot_lens(ax1, self.L5, self.f5, scale, 'L5\n')
        plot_lens(ax1, self.L7, self.f7, scale, 'L7\n')

        plot_lens(ax1, self.OBJ1, self.fobj1, scale, 'OBJ1\n')
        plot_plane(ax1, self.grid_target_front_plane, scale*0.8, ' grid target\n front')
        plot_plane(ax1, self.grid_target_back_plane, scale*6e4, ' grid target\n back')
        plot_lens(ax1, self.OBJ2, self.fobj2, scale, 'OBJ2\n')

        plot_lens(ax1, self.L9, self.f9, scale, 'L9\n')
        plot_lens(ax1, self.L10, self.f10, scale, 'L10\n')
        plot_lens(ax1, self.L11, self.f11, scale, 'L11\n')

        plot_plane(ax1, self.cam_ft_plane, 2000, 'Fourier Cam')
        plot_plane(ax1, self.cam_im_plane, 2000, 'Image Cam')

        # Plot rays
        ray2_exp_pos = self.rays[2].position_m.expand(self.rays[3].position_m.shape)
        ray2_exp = self.rays[2].copy(position_m=ray2_exp_pos)
        raylist = [ray2_exp] + self.rays[3:]
        plot_rays(ax1, raylist, fraction=0.03)

        plt.show()


# Import measurement
# cam_ft_coords_gt, cam_im_coords_gt, intensity_ft_gt, intensity_im_gt = tpm.raytrace()
matpath = 'LocalData/pencil-beam-positions/26-Feb-2021-empty/raylearn_pencil_beam_738213.520505_empty.mat'
matfile = h5py.File(matpath, 'r')

cam_size_pix = tensor((1088., 1088.)).view(1, 1, 2)
cam_ft_coords_gt = tensor((matfile['cam_ft_col'],
                           matfile['cam_ft_row'])).permute(1, 2, 0) - cam_size_pix/2 + 50
cam_im_coords_gt = tensor((matfile['cam_img_col'],
                           matfile['cam_img_row'])).permute(1, 2, 0) - cam_size_pix/2

##### Don't use image cam coords close to edge
cam_im_coords_gt[cam_im_coords_gt.abs() > cam_size_pix*0.45] = np.nan


tpm = TPM()
tpm.update()
tpm.raytrace()
# tpm.plot()

# Define Inital Guess
tpm.slm_angle = tensor((0.,), requires_grad=True)
tpm.galvo_angle = tensor((0.,), requires_grad=True)
tpm.cam_ft_shift = tensor((0., 0., 0.), requires_grad=True)
tpm.cam_im_shift = tensor((0., 0., 0.), requires_grad=True)
tpm.grid_target_zshift = tensor((0.,), requires_grad=True)
tpm.obj2_zshift = tensor((0.,), requires_grad=True)
tpm.L9_zshift = tensor((0.,), requires_grad=True)
params_obj = (tpm.obj2_zshift,)
params_other = (tpm.slm_angle,
                tpm.galvo_angle,
                tpm.cam_ft_shift,
                tpm.cam_im_shift,
                tpm.grid_target_zshift,
                tpm.L9_zshift)
tpm.update()

# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 5.0e-5, 'params': params_obj},
        {'lr': 1.0e-3, 'params': params_other},
    ], lr=1.0e-3)

iterations = 151
errors = torch.zeros(iterations)
trange = tqdm(range(iterations), desc='error: -')

#######
fig, ax = plt.subplots(figsize=(7, 7));
cam_ft_coords_gt[(norm(cam_ft_coords_gt) > 300).expand(cam_ft_coords_gt.shape)] = np.nan
not_nan_mask_ft_gt = cam_ft_coords_gt.isnan().logical_not()
for i in range(97):
    plot_coords(ax, cam_ft_coords_gt[:,i,:],
                {'color': [np.random.rand(),np.random.rand(),np.random.rand()]});
plt.show()
#######


for t in trange:
    # === Learn === #
    # Forward pass
    tpm.update()
    cam_ft_coords, cam_im_coords, intensity_ft, intensity_im = tpm.raytrace()
    #####
    cam_im_coords[cam_im_coords.abs() > cam_size_pix*0.45] = np.nan
    std_im = (cam_im_coords - cam_im_coords_gt).nan_to_num(nan=0.0).abs().std()
    cam_im_coords[(cam_im_coords - cam_im_coords_gt).nan_to_num(nan=0.0).abs() > 3*std_im] = np.nan
    #####

    # Compute and print error
    error = MSE(cam_ft_coords_gt, cam_ft_coords) \
        + MSE(cam_im_coords_gt, cam_im_coords)  # \
        #+ MSE(intensity_im_gt, intensity_im)

    error_value = error.detach().item()
    errors[t] = error_value

    # trange.desc = f'error: {error_value:<8.3g}, thick: {tpm.grid_target_thickness.detach()}'

    # error.backward(retain_graph=True)
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    if t % 25 == 0:
        fig, ax = plt.subplots(nrows=2, figsize=(5, 10))
        fig.dpi = 144

        # Plot error
        plot_coords(ax[0], cam_ft_coords_gt[44,:,:], {'label':'measured'})
        plot_coords(ax[0], cam_ft_coords[44,:,:], {'label':'sim'})
        ax[0].set_ylabel('y (pix)')
        ax[0].set_xlabel('x (pix)')
        ax[0].legend(loc=1)
        ax[0].set_title(f'Fourier Plane Cam | iter: {t}')

        plot_coords(ax[1], cam_im_coords_gt[45, :, :], {'label': 'measured'})
        plot_coords(ax[1], cam_im_coords[45, :, :], {'label': 'sim'})
        ax[1].set_ylabel('y (pix)')
        ax[1].set_xlabel('x (pix)')
        ax[1].legend(loc=1)
        ax[1].set_title(f'Image Plane Cam | iter: {t}')

        plt.show()

print(f'\n\nPrediction: {tpm.slm_angle.detach()}')

#####
cam_im_coords_gt[cam_im_coords.isnan()] = np.nan
#####


fig, ax1 = plt.subplots(figsize=(7, 7))
fig.dpi = 144

# Plot error
errorcolor = 'tab:red'
RMSEs = np.sqrt(errors.detach().cpu())
ax1.plot(RMSEs, label='error', color=errorcolor)
ax1.set_ylabel('Error (pix)')
ax1.set_ylim((0, max(RMSEs)))
ax1.legend()

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Learning parameters')
plt.show()


# %%
# from time import sleep
# for i in range(87):
#     fig, ax = plt.subplots(nrows=2, figsize=(5, 10))
#     fig.dpi = 144

#     # Plot error
#     plot_coords(ax[0], cam_ft_coords_gt[i,:,:], {'label':'measured'})
#     plot_coords(ax[0], cam_ft_coords[i,:,:], {'label':'sim'})
#     ax[0].set_ylabel('y (pix)')
#     ax[0].set_xlabel('x (pix)')
#     ax[0].legend(loc=1)
#     ax[0].set_title(f'Fourier Plane Cam | slice {i},:')

#     plot_coords(ax[1], cam_im_coords_gt[:, i, :], {'label': 'measured'})
#     plot_coords(ax[1], cam_im_coords[:, i, :], {'label': 'sim'})
#     ax[1].set_ylabel('y (pix)')
#     ax[1].set_xlabel('x (pix)')
#     ax[1].legend(loc=1)
#     ax[1].set_title(f'Image Plane Cam | slice :,{i}')

#     sleep(0.3)
#     plt.show()


# %%
xdiff_im, ydiff_im = (cam_im_coords_gt - cam_im_coords).detach().unbind(-1)
x_im_gt, y_im_gt = cam_im_coords_gt.detach().unbind(-1)

fig, ax = plt.subplots(nrows=2, figsize=(5, 10))
fig.dpi = 144

ax[0].plot(x_im_gt, xdiff_im, '.')
ax[0].set_xlabel('x_im_gt (pix)')
ax[0].set_ylabel('xdiff_im (pix)')
ax[0].set_title('Image Plane x error')

ax[1].plot(y_im_gt, ydiff_im, '.')
ax[1].set_xlabel('y_im_gt (pix)')
ax[1].set_ylabel('ydiff_im (pix)')
ax[1].set_title('Image Plane y error')

plt.show()

# %%
xdiff_im, ydiff_im = (cam_im_coords_gt - cam_im_coords).detach().unbind(-1)
x_galvo, y_galvo = tpm.galvo_rots.detach().expand(cam_im_coords_gt.shape).unbind(-1)

fig, ax = plt.subplots(nrows=2, figsize=(5, 10))
fig.dpi = 144

ax[0].plot(x_galvo, xdiff_im, '.')
ax[0].set_xlabel('x_galvo (rad)')
ax[0].set_ylabel('xdiff_im (pix)')
ax[0].set_title('Image Plane x error')

ax[1].plot(y_galvo, ydiff_im, '.')
ax[1].set_xlabel('y_galvo (rad)')
ax[1].set_ylabel('ydiff_im (pix)')
ax[1].set_title('Image Plane y error')

plt.show()


# %%
xdiff_im, ydiff_im = (cam_im_coords_gt - cam_im_coords).detach().unbind(-1)
x_slm, y_slm = tpm.slm_coords.detach().expand(cam_im_coords_gt.shape).unbind(-1)

fig, ax = plt.subplots(nrows=2, figsize=(5, 10))
fig.dpi = 144

ax[0].plot(x_slm, xdiff_im, '.')
ax[0].set_xlabel('x_slm (slm heights)')
ax[0].set_ylabel('xdiff_im (pix)')
ax[0].set_title('Image Plane x error')

ax[1].plot(y_slm, ydiff_im, '.')
ax[1].set_xlabel('y_slm (slm heights)')
ax[1].set_ylabel('ydiff_im (pix)')
ax[1].set_title('Image Plane y error')

plt.show()
