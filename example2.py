"""Example 2."""

import torch
from torch import nn, optim
from vector_functions import vector, unit
from ray_plane import Plane, CoordPlane
from optical import collimated_source, ideal_lens
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def optical_system(f):
    """Optical System."""
    # Dimensions
    N1 = 11
    N2 = 11
    vsize = (3, N1, N2)
    ssize = (1, N1, N2)
    nsize = (1, 1, 1)

    # Lens properties
    lens_position = 1e-3 * vector((0, 0, 25), vsize)
    lens_normal = unit(vector((0.3, 0, -1), vsize))
    lens_plane = Plane(lens_position, lens_normal)

    # Camera properties
    pixsize = 10e-6
    cam_position = 1e-3 * vector((0, 0, 60), vsize)
    cam_x = pixsize * unit(vector((-1, 0, 0.4), vsize))
    cam_y = pixsize * unit(vector((0, 1, 0), vsize))
    cam_plane = CoordPlane(cam_position, cam_x, cam_y)

    # Ray tracing
    rays1 = collimated_source(vsize)
    rays2 = ideal_lens(rays1, lens_plane, f)
    rays3 = rays2.intersect_plane(cam_plane)
    camcoords = cam_plane.transform(rays3)

    return camcoords


nsize = (1, 1, 1)
f_gt = vector([50e-3], nsize)                                   # Focal length

f = vector([65e-3], nsize)                                   # Focal length
f.requires_grad = True

traced_optical_system = torch.jit.trace(optical_system, f)

parameters = [f]
optimizer = torch.optim.Adam(parameters, lr=3.0e-3, betas=(0.6, 0.999))
# optimizer = torch.optim.SGD(parameters, lr=5.0e-10)
criterion = nn.MSELoss(reduction='mean')
camcoords_gt = optical_system(f_gt)

iterations = 50
losses = torch.zeros(iterations)
fpreds = torch.zeros(iterations)
trange = tqdm(range(iterations), desc='Loss: -')

for t in trange:
    # Forward pass
    camcoords = traced_optical_system(f)

    # Compute and print loss
    loss = criterion(camcoords, camcoords_gt)
    loss_value = loss.detach().item()
    losses[t] = loss_value
    fpreds[t] = f.detach().item()

    trange.desc = f'Loss: {loss_value:<8.3g}'

    optimizer.zero_grad()

    loss.backward(retain_graph=True)
    optimizer.step()


# === Plot === #
fig, ax1 = plt.subplots(figsize=(6, 6), dpi=120)
varcolor = 'tab:blue'

# Plot Ground Truth
plt.plot([0, iterations], [f_gt, f_gt], '--k', label='Ground Truth')

# Plot Loss
ax1.set_xlabel('Iteration')
ax1.set_ylabel('f ($m$)', color=varcolor)
ax1.plot(fpreds.detach().cpu(), color=varcolor, label='f')
ax1.tick_params(axis='y', labelcolor=varcolor)
plt.legend(loc=9)

# Plot Variable
ax2 = ax1.twinx()
losscolor = 'tab:red'
ax2.set_ylabel('MSE Loss ($m^2$)', color=losscolor)
plt.plot(losses.detach().cpu(), label='Loss', color=losscolor)
ax2.tick_params(axis='y', labelcolor=losscolor)
ax2.legend(loc=0)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Learning focal distance')
plt.show()
