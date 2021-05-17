"""Example 3."""

import torch
from torch import nn, optim, Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import imageio

from vector_functions import vector, unit
from ray_plane import CoordPlane
from optical import collimated_source, intensity_mask_smooth_grid
from plot_functions import plot_rays, plot_coords, ray_positions, plot_lens, plot_plane

# Use GPU
torch.set_default_tensor_type('torch.FloatTensor')


def optical_system(mask_position):
    """Optical System."""
    # Dimensions
    N1 = 25
    N2 = 25
    
    # Source properties
    src_pos = vector((0,0,0), (3,1,1))
    src_x = 1e-2 * vector((-1,0,0), (3,1,1))
    src_y = 1e-2 * vector((0,1,0), (3,1,1))
    src_plane = CoordPlane(src_pos, src_x, src_y)

    # Lens properties
    spacing_x = 10e-3
    spacing_y = 10e-3
    power = 3
    mask_x = unit(vector((-1, 0, 0), (3, 1, 1)))
    mask_y = unit(vector((0, 1, 0), (3, 1, 1)))
    mask_plane = CoordPlane(mask_position, mask_x, mask_y)

    # Camera properties
    pixsize = 10e-6
    cam_position = 1e-3 * vector((0, 0, 70), (3, 1, 1))
    cam_x = pixsize * unit(vector((-1, 0, 0), (3, 1, 1)))
    cam_y = pixsize * unit(vector((0, 1, 0), (3, 1, 1)))
    cam_plane = CoordPlane(cam_position, cam_x, cam_y)

    # Ray tracing
    rays1 = collimated_source(src_plane, N1, N2)
    rays2 = intensity_mask_smooth_grid(rays1, mask_plane, spacing_x, spacing_y, power)
    rays3 = rays2.intersect_plane(cam_plane)
    camcoords = cam_plane.transform(rays3)

    return rays3, camcoords


# Dimensions
N1 = 7
N2 = 7
vsize = (3, N1, N2)
nsize = (1, 1, 1)

# Define Ground Truth
mask_position_gt = 1e-3 * vector((0.1, 0.3, 30), (3, 1, 1))
parameters_gt = [mask_position_gt]
rays3_gt, camcoords_gt = optical_system(*parameters_gt)

# Define Inital Guess
mask_position = 1e-3 * vector((0.2, 0.15, 30), (3, 1, 1))
mask_position.requires_grad = True
parameters = [mask_position]

# Trace computational graph
# traced_optical_system = torch.jit.trace(optical_system, parameters)

# Define optimizer
optimizer = optim.Adam([
        {'lr': 1.0e-4, 'params': mask_position},
    ], lr=1.0e-3, betas=(0.9, 0.999))

criterion = nn.MSELoss(reduction='mean')

iterations = 100
errors = torch.zeros(iterations)
trange = tqdm(range(iterations), desc='error: -')


for t in trange:
    # === Learn === #
    # Forward pass
    rays3, camcoords = optical_system(*parameters)
    
    # Compute and print error
    MSE = criterion(camcoords, camcoords_gt) + criterion(rays3.intensity, rays3_gt.intensity)
    error = MSE
    error_value = error.detach().item()
    errors[t] = error_value
    
    trange.desc = f'error: {error_value:<8.3g}'

    optimizer.zero_grad()

    error.backward(retain_graph=True)
    optimizer.step()

print(f'Ground Turth: {mask_position_gt.detach()}\nPrediction: {mask_position.detach()}')


fig, ax1 = plt.subplots(figsize=(7, 7))
fig.dpi = 144

# Ray tracer plot

# Plot error
errorcolor = 'tab:red'
RMSEs = np.sqrt(errors.detach().cpu())
ax1.plot(RMSEs, label='error', color=errorcolor)
ax1.set_ylabel('Error (pix)')
ax1.set_ylim((0, max(RMSEs)))
ax1.legend()

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Learning parameters')
plt.show()
