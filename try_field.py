
import matplotlib.pyplot as plt
import numpy as np
from torch import tensor, linspace
import torch

from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays
from optical import point_source, collimated_source, ideal_lens, snells
from interpolate import interpolate2d
from field import field_from_rays, coord_grid
from vector_functions import cartesian3d

class FocusingSystem():

    def __init__(self):
        # Define coordinate system
        origin, x, y, z = cartesian3d()

        # Define source
        self.beam_width = 0.2e-3
        self.sourceplane = CoordPlane(origin, -x*self.beam_width, y*self.beam_width)
        self.source_Nx = 21
        self.source_Ny = 21

        self.f1 = 1e-3
        self.L1 = Plane(origin + self.f1 * z, -z)

        # Define camera
        self.cam_im_plane = CoordPlane(origin + 1.5*self.f1*z, -x, y*np.cos(0.2)-z*np.sin(0.2))

        self.slab1 = Plane(origin+1.2*self.f1*z,-z)
        self.slab2 = Plane(origin+1.3*self.f1*z,-z)

        self.imageplane = Plane(origin+2*self.f1*z,-z)
        
        self.n1 = 1
        self.n2 = 1.5

    def raytrace(self):
        self.rays = [collimated_source(self.sourceplane, self.source_Nx, self.source_Ny)]
        self.rays.append(ideal_lens(self.rays[-1], self.L1, self.f1))

        # self.rays.append(self.rays[-1].intersect_plane(self.slab1))
        # self.rays.append(snells(self.rays[-1], self.slab1.normal, self.n2))

        # self.rays.append(self.rays[-1].intersect_plane(self.slab2))
        # self.rays.append(snells(self.rays[-1], self.slab2.normal, self.n1))

        self.rays.append(self.rays[-1].intersect_plane(self.cam_im_plane))

        self.rays.append(self.rays[-1].intersect_plane(self.imageplane))

        return self.cam_im_plane.transform(self.rays[-1])

    def plot(self):
        fig = plt.figure(figsize=(8,4))
        fig.dpi = 144
        ax = plt.gca()
        
        # Plot lenses and planes
        scale = 2*self.beam_width
        plot_lens(ax, self.L1, self.f1, scale, '⟷ L1\n  ')
        plot_plane(ax, self.cam_im_plane, scale, ' Cam')
        plot_plane(ax, self.imageplane, scale)

        # plot_plane(ax, self.slab1, scale)
        # plot_plane(ax, self.slab2, scale)

        # Plot rays
        plot_rays(ax, self.rays)

        plt.show()

class PlaneWaveSystem:

    def __init__(self):
        # Define coordinate system
        origin, x, y, z = cartesian3d()

        # Define source
        self.theta = 0.3
        self.beam_width = 5e-6
        self.sourceplane = CoordPlane(origin, -x*self.beam_width, y*self.beam_width*np.cos(self.theta)-z*self.beam_width*np.sin(self.theta))
        self.source_Nx = 5
        self.source_Ny = 5

        self.d = 50e-6

        # Define camera
        self.cam_im_plane = CoordPlane(origin + self.d*z + self.d*np.tan(self.theta)*y, -x, y)

    def raytrace(self):
        self.rays = [collimated_source(self.sourceplane, self.source_Nx, self.source_Ny)]
        self.rays.append(self.rays[-1].intersect_plane(self.cam_im_plane))

        return self.cam_im_plane.transform(self.rays[-1])

    def plot(self):
        fig = plt.figure(figsize=(8,4))
        fig.dpi = 144
        ax = plt.gca()
        
        # Plot lenses and planes
        scale = 0.025
        plot_plane(ax, self.cam_im_plane, 0.01 * scale, ' Cam')

        # Plot rays
        plot_rays(ax, self.rays)

        plt.show()

# Simple system for testing
system = FocusingSystem()
cam_coords = system.raytrace()
system.plot()

# Interpolation
planepts = 200
max_size = 0.6*system.beam_width

field_coords = coord_grid(limits=(-max_size, max_size, -max_size,max_size), resolution=(planepts,planepts))

field_out = field_from_rays(system.rays[-2], system.cam_im_plane, field_coords)

fig = plt.figure(figsize=(5, 4))
fig.dpi = 144
ax1 = plt.gca()
plt.imshow(torch.angle(field_out), extent=(-max_size, max_size, -max_size, max_size), interpolation='none', origin='lower')
plt.colorbar()
plt.title('Field at camera plane')
plt.xlabel('x')
plt.ylabel('y')
plt.show()