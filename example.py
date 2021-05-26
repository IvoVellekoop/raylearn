"""Example."""

from torch import tensor

from vector_functions import unit
from ray_plane import Plane, CoordPlane
from optical import collimated_source, ideal_lens
from torchviz import make_dot


size = (3, 7, 7)

# Source properties
src_pos = tensor((0., 0., 0.)).requires_grad_(True)
src_x = 1e-2 * tensor((-1., 0., 0.)).requires_grad_(True)
src_y = 1e-2 * tensor((0., 1., 0.)).requires_grad_(True)
src_plane = CoordPlane(src_pos, src_x, src_y)

# Lens properties
f = 50e-3                                   # Focal length
lens_position = 1e-3 * tensor((0., 0., 25.))
lens_normal = unit(tensor((0, 0.3, -1)))
lens_plane = Plane(lens_position, lens_normal)

# Camera properties
pixsize = 10e-6
cam_position = 1e-3 * tensor((0., 0., 60.))
cam_x = pixsize * unit(tensor((-1., 0., 0.)))
cam_y = pixsize * unit(tensor((0, 1, -0.2)))
cam_plane = CoordPlane(cam_position, cam_x, cam_y)

# Ray tracing
rays = [collimated_source(src_plane, size[1], size[2])]
rays.append(ideal_lens(rays[0], lens_plane, f))
rays.append(rays[-1].intersect_plane(cam_plane))
camcoords = cam_plane.transform(rays[-1])


if True:
    # Graphs
    make_dot((rays[1].position_m, rays[1].direction), params={'Raypos': rays[0].position_m, 'Raydir': rays[0].direction})

    make_dot((rays[2].position_m, rays[2].direction), params={'Raypos': rays[0].position_m, 'Raydir': rays[0].direction})

    make_dot(camcoords, params={'Raypos': rays[0].position_m, 'Raydir': rays[0].direction})
