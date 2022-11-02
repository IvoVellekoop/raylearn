"""
Example of plotting a cylinder and saving frames. The frames can be combined into a GIF animation
with e.g. GIMP.
"""

import numpy as np
from matplotlib import pyplot as plt

from ray_plane import CoordPlane
from vector_functions import unit, cartesian3d, rotate
from optical import cylinder_interface, point_source
from plot_functions import plot_cylinder, plot_rays, plot_plane

origin, x, y, z = cartesian3d()

# Animation parameters
do_save = False
num_frames = 100
savepath = 'plots/rotating_cylinder/'

# Parameters cylinder
radius_m = 0.7
length_m = 3
offset_m = 0
n_cyl = 1.3

# Parameters point source
Nx = 1
Ny = 9
tan_opening_angle = 0.4

# Paremeters screen and medium
screen_theta = 0.2
n_medium = 1.0

# Define static planes
source_plane = CoordPlane(-2*z, tan_opening_angle * x, tan_opening_angle * y)
screen_plane = CoordPlane(2*z, rotate(-x, y, screen_theta), y)

# Define rotation angles
thetas = np.linspace(0, 2*np.pi * (num_frames-1)/num_frames, num_frames)

# Initialize plotting
fig = plt.figure(figsize=(4, 4), dpi=110)
ax = plt.gca()

# Loop over cylinder rotation angle
for i in range(num_frames):
    # Define cylinder plane
    theta = thetas[i]
    cylinder_x = rotate(unit(x+2*z), z, theta)
    cylinder_y = rotate(y, z, theta)
    cylinder_plane = CoordPlane(origin - 0.5*z, cylinder_x, cylinder_y)

    # Ray tracing
    rays = [point_source(source_plane, Nx, Ny)]
    rays += [cylinder_interface(rays[-1], cylinder_plane, radius_m, n_cyl)]
    rays += [cylinder_interface(rays[-1], cylinder_plane, radius_m, n_medium)]
    rays += [rays[-1].intersect_plane(screen_plane)]

    # Plotting
    plt.figure(fig)
    ax.clear()
    ax.set_aspect(1)

    # Plot contents
    plot_plane(ax, screen_plane, scale=1.8)
    plot_rays(ax, rays)
    plot_cylinder(ax, cylinder_plane, radius_m, length_m, offset_m)

    # Limits, labels, title
    plt.xlim((-2.5, 2.5))
    plt.ylim((-2.5, 2.5))
    plt.xlabel('z (m)')
    plt.ylabel('y (m)')
    plt.title('Propagate point source\nthrough rotating cylinder')

    # Draw and save
    plt.draw()
    if do_save:
        plt.savefig(savepath + f'rotating_cylinder_frame{i:03d}.png')
    plt.pause(1e-3)
