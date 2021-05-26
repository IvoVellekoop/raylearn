"""Example 2."""

import torch
from torch import nn, optim, tensor, Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import imageio

from vector_functions import unit
from ray_plane import Plane, CoordPlane
from optical import collimated_source, ideal_lens
from plot_functions import plot_rays, plot_coords, ray_positions, plot_lens, plot_plane

# Use GPU
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def optical_system(f, lens_normal_shift):
    """Optical System."""
    # Dimensions
    N1 = 7
    N2 = 7

    # Source properties
    src_pos = tensor((0., 0., 0.))
    src_x = 1e-2 * tensor((1., 0., 0.))
    src_y = 1e-2 * tensor((0., 1., 0.))
    src_plane = CoordPlane(src_pos, src_x, src_y)
    print(src_plane.normal)

    # Lens properties
    lens_position = 1e-3 * tensor((0., 0., 30.))
    y = tensor((0., 1., 0.))
    lens_normal = unit(tensor((0., 0., -1.)) + y*lens_normal_shift)
    lens_plane = Plane(lens_position, lens_normal)

    # Camera properties
    pixsize = 10e-6
    cam_position = 1e-3 * tensor((0., 0., 70.))
    cam_x = pixsize * unit(tensor((-1., 0., 0.)))
    cam_y = pixsize * unit(tensor((0., 1., 0.)))
    cam_plane = CoordPlane(cam_position, cam_x, cam_y)

    # Ray tracing
    rays1 = collimated_source(src_plane, N1, N2)
    rays2 = ideal_lens(rays1, lens_plane, f)
    rays3 = rays2.intersect_plane(cam_plane)
    camcoords = cam_plane.transform(rays3)

    rays = (rays1, rays2, rays3)

    return camcoords, rays, lens_plane, cam_plane


# Define Ground Truth
f_gt = Tensor([50e-3])
lens_normal_shift_gt = Tensor([0.1])
parameters_gt = [f_gt, lens_normal_shift_gt]
camcoords_gt_perfect, raylist_gt, lens_plane, cam_plane = optical_system(*parameters_gt)
stdcoords = torch.std(camcoords_gt_perfect)
camcoords_gt = camcoords_gt_perfect + 1e-2 * torch.randn(camcoords_gt_perfect.shape) * stdcoords

# Define Inital Guess
f = tensor((80e-3,))
f.requires_grad = True
lens_normal_shift = Tensor([0.25])
lens_normal_shift.requires_grad = True
parameters = [f, lens_normal_shift]

# Trace computational graph
# traced_optical_system = torch.jit.trace(optical_system, parameters)

# Define optimizer
optimizer = optim.Adam([
        {'lr': 1.0e-3, 'params': f},
        {'lr': 5.0e-3, 'params': lens_normal_shift},
    ], lr=1.0e-2, betas=(0.9, 0.999))

criterion = nn.MSELoss(reduction='mean')

iterations = 120
errors = torch.zeros(iterations)
fpreds = torch.zeros(iterations)
tiltpreds = torch.zeros(iterations)
coordpreds = []
rayspreds = []
lens_planes = []
cam_planes = []
trange = tqdm(range(iterations), desc='error: -')


for t in trange:
    # === Learn === #
    # Forward pass
    camcoords, raylist, lens_plane, cam_plane = optical_system(*parameters)
    coordpreds.append(camcoords.detach())
    rayspreds.append(raylist)
    lens_planes.append(lens_plane)
    cam_planes.append(cam_plane)

    # Compute and print error
    MSE = criterion(camcoords, camcoords_gt)
    error = MSE
    error_value = error.detach().item()
    errors[t] = error_value
    fpreds[t] = f.detach().item()
    tiltpreds[t] = lens_normal_shift.detach().item()

    trange.desc = f'error: {error_value:<8.3g}'

    optimizer.zero_grad()

    error.backward(retain_graph=True)
    optimizer.step()

# %%
with imageio.get_writer('plots/learnplot.gif', mode='I', fps=20) as writer:
    for t in tqdm(range(iterations)):
    # for t in [0]:
        # === Plot === #
        fig = plt.figure(figsize=(7, 7))
        fig.dpi = 144
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        # Ray tracer plot
        ax1.text(0, 0.012, 'Collimated rays $\\rightarrow$')
        plot_lens(ax1, lens_planes[t], fpreds[t], 0.012)
        cam_plane_plot = CoordPlane(cam_plane.position_m, cam_plane.x * 700, cam_plane.y * 700)
        plot_plane(ax1, cam_plane_plot, '')
        ax1.text(0.065, 0.008, 'Camera')
        plot_rays(ax1, rayspreds[t])
        ax1.set_title('Ray Tracer prediction')

        # Cam coords plot
        plot_coords(ax2, camcoords_gt, {'marker': '+', 'label': 'Measurement (fake)'})
        plot_coords(ax2, coordpreds[t], {'color': 'black', 'label': 'Prediction'})
        ax2.set_xlim((-700, 700))
        ax2.set_ylim((-700, 700))
        ax2.set_title('Camera coordinates')
        ax2.legend(loc=9)

        # Plot learning
        # fig, ax1 = plt.subplots(figsize=(5, 5), dpi=144)
        fcolor = 'tab:blue'
        tiltcolor = 'tab:green'
        errorcolor = 'tab:red'

        # Plot Ground Truth
        ln_f_gt    = ax3.plot([0, iterations], [f_gt]*2,
                              '--', color=fcolor, label='f Ground Truth')
        ln_tilt_gt = ax3.plot([0, iterations], [lens_normal_shift_gt]*2,
                              '--', color=tiltcolor, label='tilt point\nGround Truth')

        # Plot variable
        ln_f    = ax3.plot(fpreds.detach().cpu()[:t],    color=fcolor,    label='f')
        ln_tilt = ax3.plot(tiltpreds.detach().cpu()[:t], color=tiltcolor, label='tilt point')
        # ax3.tick_params(axis='y', labelcolor=fcolor)
        ax3.set_ylim((0, 0.25))
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('meters')

        # Plot error
        ax4 = ax3.twinx()
        ax4.set_ylabel('Error (pix)', color=errorcolor)
        RMSEs = np.sqrt(errors.detach().cpu())
        ln_RMSE = plt.plot(RMSEs[:t], label='Error', color=errorcolor)
        ax4.tick_params(axis='y', labelcolor=errorcolor)
        ax4.set_ylim((0, max(RMSEs)))

        lns = ln_f + ln_tilt + ln_f_gt + ln_tilt_gt + ln_RMSE
        labs = [ln.get_label() for ln in lns]
        ax4.legend(lns, labs, loc=0, fontsize=9)

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Learning parameters')

        filename = f'plots/learnplot-{t}.png'
        plt.savefig(filename)
        writer.append_data(imageio.imread(filename))
        plt.close()
