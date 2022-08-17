"""Example 2."""

import torch
from torch import optim, Tensor
from vector_functions import vector, unit, cartesian3d
from ray_plane import Plane, CoordPlane
from optical import collimated_source, thin_lens
import matplotlib.pyplot as plt
from tqdm import tqdm


# Set defaults
torch.set_default_tensor_type('torch.DoubleTensor')
plt.rc('font', size=12)


def optical_system(f, lens_normal_x_shift):
    """Optical System."""
    # Coordinate system
    origin, x, y, z = cartesian3d()

    # Dimensions
    beam_width = 10e-3
    source_plane = CoordPlane(origin, -x*self.beam_width, y*self.beam_width)
    source_Nx = 5
    source_Ny = 5

    # Lens properties
    lens_position = Tensor((0, 0, 25e-3))
    x = vector((1, 0, 0), (3, 1, 1))
    lens_normal = unit(vector((0, 0, -1), (3, 1, 1)) + x*lens_normal_x_shift)
    lens_plane = Plane(lens_position, unit(lens_normal))

    # Camera properties
    pixsize = 10e-6
    cam_position = 1e-3 * vector((0, 0, 60), vsize)
    cam_x = pixsize * unit(vector((-1, 0, 0.4), vsize))
    cam_y = pixsize * unit(vector((0, 1, 0), vsize))
    cam_plane = CoordPlane(cam_position, cam_x, cam_y)

    # Ray tracing
    rays1 = collimated_source(vsize)
    rays2 = thin_lens(rays1, lens_plane, f)
    rays3 = rays2.intersect_plane(cam_plane)
    camcoords = cam_plane.transform(rays3)

    return camcoords


# Dimensions
N1 = 11
N2 = 11
vsize = (3, N1, N2)
nsize = (1, 1, 1)

# Define Ground Truth
f_gt = Tensor([50e-3])
lens_normal_x_shift_gt = Tensor([0.1])
parameters_gt = [f_gt, lens_normal_x_shift_gt]
camcoords_gt = optical_system(*parameters_gt)

# Define Inital Guess
f = Tensor([80e-3])
f.requires_grad = True
lens_normal_x_shift = Tensor([0.2])
lens_normal_x_shift.requires_grad = True
parameters = [f, lens_normal_x_shift]

# Trace computational graph
traced_optical_system = torch.jit.trace(optical_system, parameters)

# Define optimizer
optimizer = torch.optim.Adam([
        {'lr': 5.0e-3, 'params': f},
        {'lr': 1.0e-2, 'params': lens_normal_x_shift},
    ], lr=1.0e-2, betas=(0.9, 0.999))

criterion = nn.MSELoss(reduction='mean')

iterations = 950
losses = torch.zeros(iterations)
fpreds = torch.zeros(iterations)
xtiltpreds = torch.zeros(iterations)
trange = tqdm(range(iterations), desc='error: -')

for t in trange:
    # Forward pass
    camcoords = traced_optical_system(*parameters)

    # Compute and print error
    MSE = criterion(camcoords, camcoords_gt)
    error = MSE + torch.sqrt(MSE)
    error_value = error.detach().item()
    losses[t] = error_value
    fpreds[t] = f.detach().item()
    xtiltpreds[t] = lens_normal_x_shift.detach().item()

    trange.desc = f'error: {error_value:<8.3g}'

    optimizer.zero_grad()

    error.backward(retain_graph=True)
    optimizer.step()


# === Plot === #
fig, ax1 = plt.subplots(figsize=(5, 5), dpi=144)
fcolor = 'tab:blue'
xtiltcolor = 'tab:green'
errorcolor = 'tab:red'

# Plot Ground Truth
plt.plot([0, iterations], [f_gt]*2, '--', color=fcolor, label='f Ground Truth')
plt.plot([0, iterations], [lens_normal_x_shift_gt]*2, '--', color=xtiltcolor, label='x tilt Ground Truth')

# Plot error
ax1.set_xlabel('Iteration')
# ax1.set_ylabel('f ($m$)', color=fcolor)
ax1.set_ylabel('meters')
ax1.plot(fpreds.detach().cpu(), color=fcolor, label='f')
ax1.plot(xtiltpreds.detach().cpu(), color=xtiltcolor, label='x tilt')
# ax1.tick_params(axis='y', labelcolor=fcolor)
plt.legend(loc=9)

# Plot Variable
ax2 = ax1.twinx()
ax2.set_ylabel('Error ($m^2$)', color=errorcolor)
plt.plot(losses.detach().cpu(), label='Error', color=errorcolor)
ax2.tick_params(axis='y', labelcolor=errorcolor)
ax2.legend(loc=0)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Learning focal distance')
plt.show()
