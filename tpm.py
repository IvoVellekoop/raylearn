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

from vector_functions import rotate
from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays, plot_coords
from optical import ideal_lens, snells, galvo_mirror, slm_segment, intensity_mask_smooth_grid
from testing import MSE

# from torchviz import make_dot

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
        All lengths in meters.

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

        self.origin = origin
        self.x = x
        self.y = y
        self.z = z

        # Galvo
        self.galvo_plane = CoordPlane(origin, x, y)
        self.galvo_rad_per_V = (np.pi/180) / 0.5

        # SLM
        self.slm_width = 10.7e-3
        self.slm_height = 10.7e-3
        self.slm_x = x * self.slm_width
        self.slm_y = y * self.slm_height
        self.slm_plane = CoordPlane(origin, self.slm_x, self.slm_y)

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
        self.sample_plane = CoordPlane(self.OBJ1.position_m + self.fobj1*z, -x, y)

        # Grid Target
        # https://www.thorlabs.com/thorproduct.cfm?partnumber=R1L3S3P
        self.grid_target_x_spacing = 50e-6
        self.grid_target_y_spacing = 50e-6
        self.grid_target_back_plane = CoordPlane(
            self.sample_plane.position_m,
            self.grid_target_x_spacing * -x,
            self.grid_target_y_spacing * y)
        self.grid_target_thickness = 0.

        # Lens planes transmission arm
        self.OBJ2 = Plane(self.sample_plane.position_m + self.fobj2*z, -z)
        self.L9 = Plane(self.OBJ2.position_m + (self.fobj2 + self.f9)*z, -z)
        self.L10 = Plane(self.L9.position_m + (self.f9 + self.f10)*z, -z)
        self.L11 = Plane(self.L10.position_m + (self.f10 + self.f11)*z, -z)

        # Camera planes
        self.cam_pixel_size = 5.5e-6
        self.cam_ft_plane = CoordPlane(self.L10.position_m + self.f10*z,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * y)
        self.cam_im_plane = CoordPlane(self.L11.position_m + self.f11*z,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * y)

    def update(self):
        """
        Update properties depending on parameters to be learned.
        """
        z = self.z
        self.grid_target_front_plane = Plane(
            self.grid_target_back_plane.position_m - self.grid_target_thickness*z, -z)

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
        self.rays = [Ray(self.origin, -self.z)]

        # Propagation to objective 1
        self.rays.append(galvo_mirror(self.rays[-1], self.galvo_plane, self.galvo_rots))
        self.rays.append(slm_segment(self.rays[-1], self.slm_plane, self.slm_coords))
        self.rays.append(ideal_lens(self.rays[-1], self.L5, self.f5))
        self.rays.append(ideal_lens(self.rays[-1], self.L7, self.f7))
        self.rays.append(ideal_lens(self.rays[-1], self.OBJ1, self.fobj1))
        self.rays.append(self.rays[-1].copy(refractive_index=self.n_water))

        # # Propagation through grid target glass slide
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

cam_size_pix = tensor((2048., 1088.)).view(1,1,2)
cam_ft_coords_gt = tensor((matfile['cam_ft_col'], \
                           matfile['cam_ft_row'])).permute(1, 2, 0) - cam_size_pix/2
cam_im_coords_gt = tensor((matfile['cam_img_col'], \
                           matfile['cam_img_row'])).permute(1, 2, 0) - cam_size_pix/2


tpm = TPM()
tpm.update()
tpm.raytrace()
tpm.plot()

# Define Inital Guess
tpm.grid_target_thickness = tensor((0.,), requires_grad=True)
tpm.update()
parameters = (tpm.grid_target_thickness,)

# Trace computational graph
# tpm.traced_raytrace = torch.jit.trace_module(tpm, {'raytrace': []})

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 2.0e-4, 'params': tpm.grid_target_thickness},
    ], lr=1.0e-3)


iterations = 1
errors = torch.zeros(iterations)
trange = tqdm(range(iterations), desc='error: -')


for t in trange:
    # === Learn === #
    # Forward pass
    tpm.update()
    cam_ft_coords, cam_im_coords, intensity_ft, intensity_im = tpm.raytrace()

    # Compute and print error
    error = MSE(cam_ft_coords_gt, cam_ft_coords) \
        + MSE(cam_im_coords_gt, cam_im_coords)  # \
        #+ criterion(intensity_im_gt, intensity_im)

    error_value = error.detach().item()
    errors[t] = error_value

    # trange.desc = f'error: {error_value:<8.3g}, thick: {tpm.grid_target_thickness.detach()}'

    error.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()

    if t % 5 == 0:
        fig, ax1 = plt.subplots(figsize=(7, 7))
        fig.dpi = 144

        # Plot error
        plot_coords(ax1, cam_im_coords_gt[:, 59, :], {'label': 'measured'})
        plot_coords(ax1, cam_im_coords[:, 59, :], {'label': 'sim'})
        # plot_coords(ax1, cam_ft_coords_gt[48,:,:], {'label':'measured'})
        # plot_coords(ax1, cam_ft_coords[48,:,:], {'label':'sim'})
        ax1.set_ylabel('y (pix)')
        ax1.set_xlabel('x (pix)')
        ax1.legend()

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Fourier Plane Cam')
        plt.show()

# print(f'\n\nGround Truth: {grid_target_thickness_gt.detach()}\n' +
#       f'Prediction: {tpm.grid_target_thickness.detach()}')


# fig, ax1 = plt.subplots(figsize=(7, 7))
# fig.dpi = 144

# # Plot error
# errorcolor = 'tab:red'
# RMSEs = np.sqrt(errors.detach().cpu())
# ax1.plot(RMSEs, label='error', color=errorcolor)
# ax1.set_ylabel('Error (pix)')
# ax1.set_ylim((0, max(RMSEs)))
# ax1.legend()

# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title('Learning parameters')
# plt.show()
