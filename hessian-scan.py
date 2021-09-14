"""
Make a parameter space scan of a simple optical system and compute Hessian matrix at optimum.
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
from optical import collimated_source, ideal_lens
from testing import MSE


torch.set_default_tensor_type('torch.DoubleTensor')

plt.rc('font', size=12)



class System4F(torch.nn.Module):
    """
    A 4f system.
    """
    def __init__(self):
        """
        Define all static properties of the system.
        """
        super().__init__()

        # Define coordinate system
        origin = tensor((0., 0., 0.))
        x = tensor((1., 0., 0.))
        y = tensor((0., 1., 0.))
        z = tensor((0., 0., 1.))
        self.coordsystem = (origin, x, y, z)

        # Define lenses
        self.f1 = 100e-3
        self.f2 = 100e-3

        # Define source
        self.beam_width = 10e-3
        self.source_plane = CoordPlane(origin, -x*self.beam_width, y*self.beam_width)
        self.source_Nx = 4
        self.source_Ny = 4

        # Define camera
        self.cam_plane = CoordPlane(origin + (2*self.f1 + 2*self.f2)*z, -x, y)

    def update(self):
        """
        Update dynamic properties.
        """
        origin, x, y, z = self.coordsystem
        self.L1 = Plane(origin + (self.f1 + self.L1_shift)*z, -z)
        self.L2 = Plane(origin + (2*self.f1 + self.f2 + self.L2_shift)*z, -z)

    def raytrace(self):
        # Source
        self.rays = [collimated_source(self.source_plane, self.source_Nx, self.source_Ny)]

        # Lenses of 4f system
        self.rays.append(ideal_lens(self.rays[-1], self.L1, self.f1))
        self.rays.append(ideal_lens(self.rays[-1], self.L2, self.f2))

        # Camera
        self.rays.append(self.rays[-1].intersect_plane(self.cam_plane))
        self.cam_coords = self.cam_plane.transform(self.rays[-1])

        return self.cam_coords

    def plot(self):
        """Plot the 4f system and the rays."""
        fig = plt.figure(figsize=(9, 4))
        fig.dpi = 144
        ax1 = plt.gca()

        # Plot lenses and planes
        scale = 0.025
        plot_lens(ax1, self.L1, self.f1, scale, '⟷ L1\n  ')
        plot_lens(ax1, self.L2, self.f2, scale, '⟷ L2\n  ')
        plot_plane(ax1, self.cam_plane, scale, ' Cam')

        # Plot rays
        plot_rays(ax1, self.rays)

        plt.show()


system4f = System4F()
shift1_gt = 0
shift2_gt = 0
system4f.L1_shift = shift1_gt
system4f.L2_shift = shift2_gt
system4f.update()
cam_coords_gt = system4f.raytrace()
system4f.plot()

N_shifts = 50
shift_min = -75e-3
shift_max =  75e-3
system4f.L1_shift = torch.linspace(shift_min, shift_max, N_shifts).view(1, -1, 1, 1, 1)
system4f.L2_shift = torch.linspace(shift_min, shift_max, N_shifts).view(-1, 1, 1, 1, 1)
system4f.update()
cam_coords = system4f.raytrace()

RMSE = ((cam_coords_gt - cam_coords)**2).mean(dim=(2, 3, 4)).sqrt()

# Plot scan
fig = plt.figure(figsize=(5, 4))
fig.dpi = 144
ax1 = plt.gca()
plt.imshow(RMSE, origin='lower', extent=(shift_min, shift_max, shift_min, shift_max))
plt.colorbar()
plt.contour(system4f.L1_shift.view(-1), system4f.L2_shift.view(-1), RMSE, levels=30, colors='k')
plt.plot(shift1_gt, shift2_gt, '.', color='white')
plt.text(shift1_gt, shift2_gt, ' Ground Truth', color='lightgrey')
plt.title('RMSE (m) - Parameter scan')
plt.xlabel('Lens 1 shift (m)')
plt.ylabel('Lens 2 shift (m)')
plt.show()
