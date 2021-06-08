"""
The Two Photon Microscope with a Grid Target.

Fit measurements of an absorbing Grid Target in the Two Photon Microscope.
"""

import torch
from torch import tensor, stack, meshgrid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from vector_functions import rotate
from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays, plot_coords
from optical import ideal_lens, snells, galvo_mirror, slm_segment, intensity_mask_smooth_grid


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
        self.n_water = 1.3304
        self.n_target = 1.5191

        # Define coordinate system
        origin = tensor((0., 0., 0.))
        x = tensor((1., 0., 0.))
        y = tensor((0., 1., 0.))
        z = tensor((0., 0., 1.))

        # Initial Ray
        self.ray0 = Ray(origin, -z)

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
        # Note: Create fake data for the time being
        SX = 5
        slm_coords_lin = torch.linspace(-0.4, 0.4, SX)
        self.slm_coords = stack(meshgrid(slm_coords_lin, slm_coords_lin), -1).view(SX, SX, 1, 1, 2)

        GX = 4
        galvo_volt_lin = torch.linspace(-0.02, 0.02, GX)
        galvo_rots_lin = galvo_volt_lin * self.galvo_rad_per_V
        self.galvo_rots = stack(meshgrid(galvo_rots_lin, galvo_rots_lin), -1).view(GX, GX, 2)

        # Focal distances (m)
        self.f5 = 150e-3
        self.f7 = 300e-3
        self.f9 = 150e-3            #### Dubble check
        self.f10 = 200e-3           #### Dubble check
        self.f11 = 100e-3           #### Dubble check
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
        self.grid_target_thickness = 1.5e-3  ###
        self.grid_target_x_spacing = 50e-6
        self.grid_target_y_spacing = 50e-6
        self.grid_target_front_plane = CoordPlane(
            self.sample_plane.position_m - self.grid_target_thickness*z,
            self.grid_target_x_spacing * -x,
            self.grid_target_y_spacing * y)

        # Lens planes transmission arm
        self.OBJ2 = Plane(self.sample_plane.position_m + self.fobj2*z, -z)
        self.L9 = Plane(self.OBJ2.position_m + (self.fobj2 + self.f9)*z, -z)
        self.L10 = Plane(self.L9.position_m + (self.f9 + self.f10)*z, -z)
        self.L11 = Plane(self.L10.position_m + (self.f10 + self.f11)*z, -z)

        # Camera planes
        self.cam_ft_plane = CoordPlane(self.L10.position_m + self.f10*z, -x, y)
        self.cam_im_plane = CoordPlane(self.L11.position_m + self.f11*z, -x, y)

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
        # Propagation up to objective 1
        self.ray1 = galvo_mirror(self.ray0, self.galvo_plane, self.galvo_rots)
        self.ray2 = slm_segment(self.ray1, self.slm_plane, self.slm_coords)
        self.ray3 = ideal_lens(self.ray2, self.L5, self.f5)
        self.ray4 = ideal_lens(self.ray3, self.L7, self.f7)
        self.ray5 = ideal_lens(self.ray4, self.OBJ1, self.fobj1)
        self.ray6 = self.ray5.copy(refractive_index=self.n_water)

        # Propagation between objectives
        self.ray7 = self.ray6.intersect_plane(self.grid_target_front_plane)
        self.ray8 = snells(self.ray7, self.grid_target_front_plane.normal, self.n_target)
        self.ray9 = self.ray8.intersect_plane(self.sample_plane)
        #### properties
        self.ray10 = intensity_mask_smooth_grid(self.ray9, self.sample_plane, 50e-3, 50e-3, 4)
        self.ray11 = snells(self.ray10, self.sample_plane.normal, 1)
        self.ray12 = ideal_lens(self.ray11, self.OBJ2, self.fobj2)

        # Propagation after objective 2
        self.ray13 = ideal_lens(self.ray12, self.L9, self.f9)
        self.ray14 = ideal_lens(self.ray13, self.L10, self.f10)
        self.ray15 = self.ray14.intersect_plane(self.cam_ft_plane)
        self.ray16 = ideal_lens(self.ray14, self.L11, self.f11)
        self.ray17 = self.ray16.intersect_plane(self.cam_im_plane)

        # Cameras
        self.cam_ft_coords = self.cam_ft_plane.transform(self.ray15)
        self.cam_im_coords = self.cam_im_plane.transform(self.ray17)

        return self.cam_ft_coords, self.cam_im_coords

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
        plot_plane(ax1, self.grid_target_front_plane, scale*1e5, 'grid\ntarget')
        plot_plane(ax1, self.sample_plane, scale*0.5, ' sample\n plane')
        plot_lens(ax1, self.OBJ2, self.fobj2, scale, 'OBJ2\n')

        plot_lens(ax1, self.L9, self.f9, scale, 'L9\n')
        plot_lens(ax1, self.L10, self.f10, scale, 'L10\n')
        plot_lens(ax1, self.L11, self.f11, scale, 'L11\n')

        plot_plane(ax1, self.cam_ft_plane, scale, 'Fourier Cam')
        plot_plane(ax1, self.cam_im_plane, scale, 'Image Cam')

        # Plot rays
        ray2_exp_pos = self.ray2.position_m.expand(self.ray3.position_m.shape)
        ray2_exp = self.ray2.copy(position_m=ray2_exp_pos)
        raylist = [ray2_exp, self.ray3, self.ray4, self.ray5, self.ray6,
                  self.ray7, self.ray8, self.ray9, self.ray10, self.ray11,
                  self.ray12, self.ray13, self.ray14, self.ray15,
                  self.ray16, self.ray17]
        plot_rays(ax1, raylist)

        plt.show()


tpm = TPM()
cam_ft_coords, cam_im_coords = tpm.raytrace()
tpm.plot()


fig = plt.figure(figsize=(15, 4))
fig.dpi = 144
ax1 = plt.gca()
plot_coords(ax1, cam_ft_coords, {'color': 'black', 'label': 'Prediction'})
plt.show()
