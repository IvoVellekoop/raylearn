"""
Make a parameter space scan of a simple optical system and compute Hessian matrix at optimum.
"""

import torch
from torch import tensor
import matplotlib.pyplot as plt
import numpy as np

from vector_functions import cartesian3d, unit, rotate
from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays
from optical import point_source, ideal_lens


# Set defaults
torch.set_default_tensor_type('torch.DoubleTensor')
plt.rc('font', size=12)


class System(torch.nn.Module):
    """
    Point sources around focal plane.
    """
    def __init__(self):
        """
        Define all static properties of the system.
        """
        super().__init__()

        # Define focal distance
        self.obj1_tubelength = 200e-3           # Objective standard tubelength
        self.obj1_magnification = 16            # Objective magnification
        self.fobj1 = self.obj1_tubelength / self.obj1_magnification

        # Define objective NA and refractive index of water
        NA = 0.8  / np.sqrt(2)
        self.n_water = 1.33
        self.obj1_radius = self.fobj1 * np.tan(np.arcsin(NA/self.n_water))
        # NA = 0.8
        # self.obj1_radius = self.fobj1 * np.tan(np.arcsin(NA))
        self.halfFOV = 0.5e-3   / np.sqrt(2)

        # Define distance to end_plane
        self.end_distance = 0.05

        # Define coordinate system
        self.coordsystem = cartesian3d()
        origin, x, y, z = self.coordsystem

        # Define objective position
        self.OBJ1 = CoordPlane(self.fobj1*z, -x*self.obj1_radius, y*self.obj1_radius)

        # Define source coordinates
        N = 21     # Start points per dim
        M = 21   # On-obj points per dim
        self.src_positions = self.halfFOV * x * torch.linspace(-1.0, 1.0, N).view(N, 1, 1) \
                           + self.halfFOV * y * torch.linspace(-1.0, 1.0, N).view(1, N, 1)
        positions_on_obj = self.OBJ1.x * torch.linspace(-1.0, 1.0, M).view(M, 1, 1, 1, 1) \
                        + self.OBJ1.y * torch.linspace(-1.0, 1.0, M).view(1, M, 1, 1, 1) \
                        + self.OBJ1.position_m
        self.directions = unit(positions_on_obj - self.src_positions)

    def raytrace(self):
        """
        Raytrace simulation through optical system.
        """
        # Source
        self.rays = [Ray(self.src_positions, self.directions, refractive_index=self.n_water)]

        # Objective
        self.rays.append(ideal_lens(self.rays[-1], self.OBJ1, self.fobj1))
        self.rays.append(self.rays[-1].copy(refractive_index=1))

        # End plane
        end_planes_positions = self.OBJ1.position_m + self.end_distance * self.rays[-1].direction
        end_planes = Plane(end_planes_positions, self.rays[-1].direction)
        self.rays.append(self.rays[-1].intersect_plane(end_planes))

        # Compute average pathlength error
        pathlengths = self.rays[-1].pathlength_m
        return pathlengths

    def plot(self):
        """Plot the system and the rays."""
        fig = plt.figure(figsize=(9, 4))
        fig.dpi = 144
        ax1 = plt.gca()

        origin, x, y, z = self.coordsystem
        viewplane = CoordPlane(origin, z, y)
        scale = 1.2

        # Plot lenses and planes
        plot_lens(ax1, self.OBJ1, self.fobj1, scale, ' OBJ1')

        # Plot rays
        plot_rays(ax1, self.rays)

        plt.xlabel('z (m)')
        plt.ylabel('y (m)')

        plt.show()


system = System()
pathlengths = system.raytrace()

# system.plot()


# # Circle mask
# def circlemask(A, B):
#     x = torch.linspace(-1.0, 1.0, N).view(1, A)
#     y = torch.linspace(-1.0, 1.0, M).view(B, 1)
#     return x*x + y*y <= 1
#
#
# Nx, Ny, Mx, My, D = pathlengths.shape
# pathlengths.sum(dim=[0, 1]) / sum(tuple(circlemask(Nx, Ny)))

pathlength_error_img = pathlengths.std(dim=[0, 1]).squeeze()
print(f'Average pathlength error: {pathlength_error_img.mean().item() * 1e6:.1f} um\n')


fig = plt.figure(figsize=(5, 4))
fig.dpi = 144
ax1 = plt.gca()
halfFOV_mm = system.halfFOV * 1e6
plt.imshow(pathlength_error_img * 1e6, origin='lower',
           extent=(-halfFOV_mm, halfFOV_mm, -halfFOV_mm, halfFOV_mm))
plt.colorbar()
plt.title('Thin lens pathlength error (um)\nof point sources at focal plane')
plt.xlabel('Focal plane x (um)')
plt.ylabel('Focal plane y (um)')
plt.show()
