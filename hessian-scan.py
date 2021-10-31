"""
Make a parameter space scan of a simple optical system and compute Hessian matrix at optimum.
"""

import torch
from torch import tensor
from torch.autograd.functional import hessian
import matplotlib.pyplot as plt

from ray_plane import Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays
from optical import collimated_source, ideal_lens


# Set defaults
torch.set_default_tensor_type('torch.DoubleTensor')
plt.rc('font', size=12)


class System4F(torch.nn.Module):
    """
    A 4f system.
    """
    def __init__(self):
        """
        Define all static properties of the system.
        """
        super().__init__()

        # Define coordinate system
        origin = tensor((0., 0., 0.))
        x = tensor((1., 0., 0.))
        y = tensor((0., 1., 0.))
        z = tensor((0., 0., 1.))
        self.coordsystem = (origin, x, y, z)

        # Define focal distances
        self.f1 = 100e-3
        self.f2 = 100e-3
        self.f3 = 100e-3

        # Define static Fourier camera lens
        self.L3 = Plane(origin + (2*self.f1 + 2*self.f2 + self.f3)*z, -z)

        # Define source
        self.beam_width = 10e-3
        self.source_plane = CoordPlane(origin, -x*self.beam_width, y*self.beam_width)
        self.source_Nx = 4
        self.source_Ny = 4

        # Define camera
        self.cam_im_plane = CoordPlane(origin + (2*self.f1 + 2*self.f2)*z, -x, y)
        self.cam_ft_plane = CoordPlane(origin + 2*(self.f1 + self.f2 + self.f3)*z, -x, y)

    def update(self):
        """
        Update dynamic properties.
        """
        origin, x, y, z = self.coordsystem
        self.L1 = Plane(origin + (self.f1 + self.L1_shift)*z, -z)
        self.L2 = Plane(origin + (2*self.f1 + self.f2 + self.L2_shift)*z, -z)

    def raytrace(self):
        """
        Raytrace simulation through optical system.
        """
        # Source
        self.rays = [collimated_source(self.source_plane, self.source_Nx, self.source_Ny)]

        # Lenses of 4f system
        self.rays.append(ideal_lens(self.rays[-1], self.L1, self.f1))
        self.rays.append(ideal_lens(self.rays[-1], self.L2, self.f2))

        # Cameras
        self.rays.append(self.rays[-1].intersect_plane(self.cam_im_plane))
        self.cam_im_coords = self.cam_im_plane.transform(self.rays[-1])
        self.rays.append(ideal_lens(self.rays[-1], self.L3, self.f3))
        self.rays.append(self.rays[-1].intersect_plane(self.cam_ft_plane))
        self.cam_ft_coords = self.cam_ft_plane.transform(self.rays[-1])

        return self.cam_im_coords, self.cam_ft_coords

    def shiftMSE(self, L1_shift, L2_shift):
        """
        Compute Mean Square Error originating from shifted lenses.
        """
        # Compute ground truth
        self.L1_shift = self.L1_shift_gt
        self.L2_shift = self.L2_shift_gt
        self.update()
        cam_im_coords_gt, cam_ft_coords_gt = self.raytrace()

        # Compute RMSE
        self.L1_shift = L1_shift
        self.L2_shift = L2_shift
        self.update()
        cam_im_coords, cam_ft_coords = self.raytrace()
        MSE = ((cam_im_coords_gt - cam_im_coords)**2).mean() \
            + ((cam_ft_coords_gt - cam_ft_coords)**2).mean()
        return MSE

    def plot(self):
        """Plot the 4f system and the rays."""
        fig = plt.figure(figsize=(9, 4))
        fig.dpi = 144
        ax1 = plt.gca()

        # Plot lenses and planes
        scale = 0.025
        plot_lens(ax1, self.L1, self.f1, scale, '⟷ L1\n  ')
        plot_lens(ax1, self.L2, self.f2, scale, '⟷ L2\n  ')
        plot_lens(ax1, self.L3, self.f3, scale, 'L3\n  ')
        plot_plane(ax1, self.cam_im_plane, scale, ' Cam')
        plot_plane(ax1, self.cam_ft_plane, scale, ' Cam')

        # Plot rays
        plot_rays(ax1, self.rays)

        plt.show()


doplot = False

# Initialize system with ground truth
system4f = System4F()
system4f.L1_shift_gt = 0
system4f.L2_shift_gt = 0
system4f.L1_shift = system4f.L1_shift_gt
system4f.L2_shift = system4f.L2_shift_gt

# Raytrace ground truth and plot
system4f.update()
cam_im_coords_gt, cam_ft_coords_gt = system4f.raytrace()
if doplot:
    system4f.plot()

# Set lens shifts
N_shifts = 50
shift_min = -75e-3
shift_max =  75e-3
system4f.L1_shift = torch.linspace(shift_min, shift_max, N_shifts).view(1, -1, 1, 1, 1)
system4f.L2_shift = torch.linspace(shift_min, shift_max, N_shifts).view(-1, 1, 1, 1, 1)

# Raytrace with lens shifts
system4f.update()
cam_im_coords, cam_ft_coords = system4f.raytrace()
RMSE = (((cam_im_coords_gt - cam_im_coords)**2).mean(dim=(2, 3, 4))
      + ((cam_ft_coords_gt - cam_ft_coords)**2).mean(dim=(2, 3, 4))).sqrt()

# Plot scan of L1 and L2 shifts
if doplot:
    fig = plt.figure(figsize=(5, 4))
    fig.dpi = 144
    ax1 = plt.gca()
    plt.imshow(RMSE, origin='lower', extent=(shift_min, shift_max, shift_min, shift_max))
    plt.colorbar()
    plt.contour(system4f.L1_shift.view(-1), system4f.L2_shift.view(-1), RMSE, levels=30, colors='k')
    plt.plot(system4f.L1_shift_gt, system4f.L2_shift_gt, '.', color='white')
    plt.text(system4f.L1_shift_gt, system4f.L2_shift_gt, ' Ground Truth', color='lightgrey')
    plt.title('RMSE (m) - Parameter scan')
    plt.xlabel('Lens 1 shift (m)')
    plt.ylabel('Lens 2 shift (m)')

    # Compute hessian matrix
    hessian_x = tensor((0.,))
    hessian_y = tensor((0.,))
    hessian_matrix = tensor(hessian(system4f.shiftMSE, (hessian_x, hessian_y)))

    eigen = torch.linalg.eigh(hessian_matrix)
    print(eigen)
    eigenvectors = eigen.eigenvectors
    eigenvalues = eigen.eigenvalues

    plt.arrow(hessian_x, hessian_y, eigenvectors[0, 0]*0.2*shift_max, eigenvectors[0, 1]*0.2*shift_max)
    plt.arrow(hessian_x, hessian_y, eigenvectors[1, 0]*0.2*shift_max, eigenvectors[1, 1]*0.2*shift_max)
    plt.text(hessian_x + eigenvectors[0, 0]*0.2*shift_max, hessian_y + eigenvectors[0, 1]*0.2*shift_max,
             f'  {eigenvalues[0]:.3f}', color='tab:blue', verticalalignment='center')
    plt.text(hessian_x + eigenvectors[1, 0]*0.2*shift_max, hessian_y + eigenvectors[1, 1]*0.2*shift_max,
             f'  {eigenvalues[1]:.3f}', color='tab:blue', verticalalignment='center')

    plt.show()

