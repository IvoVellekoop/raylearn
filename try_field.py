
import matplotlib.pyplot as plt
from torch import tensor, linspace
import torch

from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays
from optical import point_source, collimated_source, ideal_lens 
from interpolate import interpolate2d
from field import pathlength_to_phase

class FocusingSystem():

    def __init__(self):
        # Define coordinate system
        origin = tensor((0., 0., 0.))
        x = tensor((1., 0., 0.))
        y = tensor((0., 1., 0.))
        z = tensor((0., 0., 1.))

        # Define source
        self.beam_width = 10e-3
        self.sourceplane = CoordPlane(origin, -x*self.beam_width, y*self.beam_width)
        self.source_Nx = 4
        self.source_Ny = 4

        self.f1 = 100e-3
        self.L1 = Plane(origin + self.f1 * z, -z)

        # Define camera
        self.cam_im_plane = CoordPlane(origin + 1.8*self.f1*z, -x, y)

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


# Simple system for testing
system = FocusingSystem()
cam_coords = system.raytrace()
system.plot()
print(system.rays[-1].pathlength_m)

# Interpolation
planepts = 20
x_array = system.cam_im_plane.x * linspace(-0.01, 0.01, planepts).view(planepts, 1, 1)
y_array = system.cam_im_plane.y * linspace(-0.01, 0.01, planepts).view(1, planepts, 1)
field_coords = x_array + y_array
# Wat hier fout gaat: field_coords is nu 20x20x3 ipv 20x20x2. 
field_coords = torch.narrow(field_coords, 2, 0, 2) # Maken van field_coords kan vast netter dan dit

values_out_unfolded, mask = interpolate2d(cam_coords, system.rays[-1].pathlength_m, field_coords)

field = pathlength_to_phase(values_out_unfolded, 1e-6)
field_out = sum(mask * field, (0, 1, 2))

print