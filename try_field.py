
import matplotlib.pyplot as plt
import numpy as np
from torch import tensor, linspace
import torch

from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays
from optical import point_source, collimated_source, ideal_lens 
from interpolate import interpolate2d
from field import field_from_rays, coord_grid

class FocusingSystem():

    def __init__(self):
        # Define coordinate system
        origin = tensor((0., 0., 0.))
        x = tensor((1., 0., 0.))
        y = tensor((0., 1., 0.))
        z = tensor((0., 0., 1.))

        # Define source
        self.beam_width = 0.1e-3
        self.sourceplane = CoordPlane(origin, -x*self.beam_width, y*self.beam_width)
        self.source_Nx = 21
        self.source_Ny = 21

        self.f1 = 1e-3
        self.L1 = Plane(origin + self.f1 * z, -z)

        # Define camera
        self.cam_im_plane = CoordPlane(origin + 1.5*self.f1*z, -x, y)

    def raytrace(self):
        self.rays = [collimated_source(self.sourceplane, self.source_Nx, self.source_Ny)]
        self.rays.append(ideal_lens(self.rays[-1], self.L1, self.f1))
        self.rays.append(self.rays[-1].intersect_plane(self.cam_im_plane))

        return self.cam_im_plane.transform(self.rays[-1])

    def plot(self):
        fig = plt.figure(figsize=(8,4))
        fig.dpi = 144
        ax = plt.gca()
        
        # Plot lenses and planes
        scale = 0.025
        plot_lens(ax, self.L1, self.f1, scale, '⟷ L1\n  ')
        plot_plane(ax, self.cam_im_plane, scale, ' Cam')

        # Plot rays
        plot_rays(ax, self.rays)

        plt.show()

class PlaneWaveSystem:

    def __init__(self):
        # Define coordinate system
        origin = tensor((0., 0., 0.))
        x = tensor((1., 0., 0.))
        y = tensor((0., 1., 0.))
        z = tensor((0., 0., 1.))

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
planepts = 120
max_size = 0.6*system.beam_width

field_coords = coord_grid(limits=(-max_size, max_size, -max_size,max_size), resolution=(planepts,planepts))

field_out = field_from_rays(system.rays[-1], system.cam_im_plane, field_coords)

fig = plt.figure(figsize=(5, 4))
fig.dpi = 144
ax1 = plt.gca()
plt.imshow(torch.angle(field_out), extent=(-max_size, max_size, -max_size, max_size))
plt.colorbar()
plt.title('Field at camera plane')
plt.xlabel('x')
plt.ylabel('y')
plt.show()