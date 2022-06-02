"""Example: One Lens System"""

import torch
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from ray_plane import Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays, plot_coords
from optical import collimated_source, ideal_lens
from vector_functions import cartesian3d, rotate
from testing import MSE


# Set defaults
torch.set_default_tensor_type('torch.DoubleTensor')
plt.rc('font', size=12)


class SystemOneLens():
    def update(self):
        """Define/Update system parameters"""
        # Coordinate system
        origin, x, y, z = cartesian3d()

        # Collimated source
        self.beamwidth = 10e-3
        self.source_plane = CoordPlane(origin, self.beamwidth*x, self.beamwidth*y)

        # Lens
        self.lens_plane = Plane(50e-3*z, rotate(-z, x, self.theta))

        # Camera
        self.cam_plane = CoordPlane(100e-3*z, x, y)

    def raytrace(self):
        """Perform raytracing through system"""
        self.rays = [collimated_source(self.source_plane, 7, 7)]
        self.rays.append(ideal_lens(self.rays[-1], self.lens_plane, self.f1))
        self.rays.append(self.rays[-1].intersect_plane(self.cam_plane))
        self.cam_coords = self.cam_plane.transform_rays(self.rays[-1])
        return self.cam_coords

    def plot(self):
        fig = plt.figure(dpi=100)
        ax = plt.gca()

        # Plot lens, cam and rays
        plot_lens(ax, self.lens_plane, self.f1, scale=2e-2)
        plot_plane(ax, self.cam_plane, scale=2e-2)
        plot_rays(ax, self.rays)
        plt.show()


system = SystemOneLens()

# Define Ground Truth
f1_gt = 75e-3
theta_gt = 0.15
system.f1 = f1_gt
system.theta = theta_gt
system.update()
camcoords_gt = system.raytrace()

# Set initial guess
system.f1 = Tensor((100e-3,))
system.theta = Tensor((0.1,))
system.update()
system.raytrace()
system.plot()

# Define parameters
parameters = (system.f1, system.theta)
for param in parameters:
    param.requires_grad = True

# Define optimizer
optimizer = torch.optim.Adam(parameters, lr=1e-2)

iterations = 500
RMSEs = torch.zeros(iterations) * np.nan
fpreds = torch.zeros(iterations) * np.nan
thetapreds = torch.zeros(iterations) * np.nan
trange = tqdm(range(iterations), desc='error: -')


# Prepare plot
fig, ax1 = plt.subplots(figsize=(5, 5), dpi=100)
ax2 = ax1.twinx()

for t in trange:
    # Forward pass
    system.update()
    camcoords = system.raytrace()
    RMSE = MSE(camcoords, camcoords_gt)

    RMSEs[t] = RMSE.detach().item()
    fpreds[t] = system.f1.detach().item()
    thetapreds[t] = system.theta.detach().item()

    trange.desc = f'error: {RMSE.detach().item():<8.3g}'

    optimizer.zero_grad()
    RMSE.backward(retain_graph=False)
    optimizer.step()


    # === Plot === #
    if t % 10 == 0:
        ax1.clear()
        ax2.clear()

        fcolor = 'tab:blue'
        thetacolor = 'tab:green'
        errorcolor = 'tab:red'

        # Plot Ground Truth
        ax1.plot([0, iterations], [f1_gt]*2, '--', color=fcolor, label='f Ground Truth')
        ax1.plot([0, iterations], [theta_gt]*2, '--', color=thetacolor, label='$\\theta$ Ground Truth')

        # Plot error
        ax1.plot(fpreds, color=fcolor, label='f')
        ax1.plot(thetapreds, color=thetacolor, label='$\\theta$')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('meters | rad')
        ax1.legend(loc=7)

        # Plot Variable
        ax2.set_ylabel('Error (m)', color=errorcolor)
        ax2.plot(RMSEs, label='Error', color=errorcolor)
        ax2.tick_params(axis='y', labelcolor=errorcolor)
        ax2.legend(loc=2)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Learning focal distance')
        plt.draw()
        plt.pause(1e-2)

plt.show()
