"""Example."""

import torch
from vector_functions import vector, unit
from ray_plane import Plane, CoordPlane
from optical import collimated_source, ideal_lens
import matplotlib.pyplot as plt
from torchviz import make_dot


size = (3,7,7)

# Lens properties
f = 50e-3                                   # Focal length
lens_position = 1e-3 * vector((0, 0, 25), size)
lens_normal =     unit(vector((0.3, 0, -1), size))
lens_plane = Plane(lens_position, lens_normal)

# Camera properties
pixsize = 10e-6;
cam_position =  1e-3 * vector(( 0, 0, 60), size)
cam_x = pixsize * unit(vector((-1, 0, 0.4), size))
cam_y = pixsize * unit(vector(( 0, 1, 0), size))
cam_plane = CoordPlane(cam_position, cam_x, cam_y)

# Ray tracing
rays1 = collimated_source(size)
rays2 = ideal_lens(rays1, lens_plane, f)
# ray_positions_on_cam = rays2.intersect_plane(cam_plane)
# rays3 = Ray(ray_positions_on_cam, rays2.direction)
rays3 = rays2.intersect_plane(cam_plane)
camcoords = cam_plane.transform(rays3)


# Plot
points = (rays1.position_m, rays2.position_m, rays3.position_m)
dimhori=2
dimvert=0

# Prepare variables
dimlabels = ['x (m)', 'y (m)', 'z (m)']
points_hori = torch.stack(points)[:,dimhori,:,:].view(len(points), -1).detach().cpu()
points_vert = torch.stack(points)[:,dimvert,:,:].view(len(points), -1).detach().cpu()

# Plot figure
fig = plt.figure(figsize=(6,6))
fig.dpi = 120
plt.plot(points_hori, points_vert, '.-')
plt.gca().axis('equal')
plt.xlabel(dimlabels[dimhori])
plt.ylabel(dimlabels[dimvert])
plt.title('Ray Paths')
plt.show()


# Plot coords on camera
fig = plt.figure(figsize=(6,6))
fig.dpi = 120
plt.plot(camcoords[0].detach().cpu(), camcoords[1].detach().cpu(), '.', color='tab:blue')
plt.gca().axis('equal')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Camera coordinates')
plt.show()


if False:
    # Graphs
    make_dot((rays2.position_m, rays2.direction), \
             params={'Raypos':rays1.position_m, 'Raydir':rays1.direction})

    make_dot((rays3.position_m, rays3.direction), \
         params={'Raypos':rays1.position_m, 'Raydir':rays1.direction})

    make_dot(camcoords, params={'Raypos':rays1.position_m, 'Raydir':rays1.direction})
