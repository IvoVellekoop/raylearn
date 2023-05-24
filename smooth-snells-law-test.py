# Smooth Snell's law test
# Test a derived version of Snell's law based on an angle distribution, rather than a single angle

import torch
from torch import tensor
import numpy as np
import matplotlib.pyplot as plt

from vector_functions import ensure_tensor, cartesian3d
from optical import snells, snells_gaussian, snells_softplus, point_source, CoordPlane
from plot_functions import plot_plane, plot_rays


torch.set_default_tensor_type('torch.DoubleTensor')
# torch.set_default_tensor_type('torch.FloatTensor')


def snells_2d(Dxin, n1, n2):
    return n1/n2*Dxin


def arccot(ratio):
    """arccot function"""
    return np.pi/2 - np.arctan(ratio)


def smooth_snells_lorentzian_2d(DxInCenter, width, n1, n2):
    """
    Smooth Snells law (2D) based on Lorentzian sin(angle) distribution
    """
    if n2 > n1:
        total_weight = (np.arctan((1-DxInCenter)/width) + np.arctan((1 + DxInCenter)/width)) / np.pi
        DxOutMean = \
            n1/n2 * (DxInCenter + width/(2*np.pi * total_weight) * (
                np.log((DxInCenter-1)**2 + width**2)
              - np.log((DxInCenter+1)**2 + width**2)))
        return DxOutMean

    else:
        # total_weight = (arccot(n1*width / (n2-n1*DxInCenter))
        #               + arccot(n1*width / (n2+n1*DxInCenter))) / np.pi
        # DxOutMean = \
        #     n1/n2 * (DxInCenter + width/(2*np.pi * total_weight) * (
        #         np.log((n2-n1*DxInCenter)**2 / n1**2 + width**2)
        #       - np.log((n2+n1*DxInCenter)**2 / n1**2 + width**2)))
        # return DxOutMean

        return (n1 * width *
            ((DxInCenter * (arccot((n1*width)/((-DxInCenter)*n1 + n2)) +
            arccot((n1*width)/(DxInCenter*n1 + n2))))/width +
            (1/2)*(np.log(((-DxInCenter)*n1 + n2)**2/n1**2 + width**2) -
            np.log((DxInCenter*n1 + n2)**2/n1**2 + width**2)))) / \
            (n2*(arccot((n1*width)/((-DxInCenter)*n1 + n2)) +
            arccot((n1*width)/(DxInCenter*n1 + n2))))


def smooth_snells(DxInCenter, std, n1, n2):
    """
    Smoothed Snell's law, with gaussian distribution as input.

    Input
    -----
    DxInCenter
    std       Standard deviation (sigma) of gaussian distribution in k_xy plane.
    """
    DxInCrit = ensure_tensor(n2/n1)
    intensity = (torch.erf((DxInCrit + DxInCenter) / (np.sqrt(2)*std)) +
                 torch.erf((DxInCrit - DxInCenter) / (np.sqrt(2)*std))) / 2
    DxInMean = DxInCenter + np.sqrt(2/np.pi) * std / (2*intensity) * (
        - torch.exp(-((DxInCenter - DxInCrit)**2 / (2*std**2)))
        + torch.exp(-((DxInCenter + DxInCrit)**2 / (2*std**2))))
    DxOutMean = DxInMean * n1/n2
    return DxOutMean

    # Add check for extreme exponents with poor precision


N = 800
DxInCenter = torch.linspace(-1, 1, N).view((-1, 1))

# width = tensor((0.2, 0.1, 0.05, 0.02, 0.01)).view((1, -1))
stds = tensor((0.0892, 0.04, 0.02, 0.01)).view((1, -1))
n1 = 1.5
n2 = 1.3

# Plot DxOut vs DxIn
widthlabels = ['std=' + f'{w:.3f}' for w in stds.view(-1)]
plt.plot(DxInCenter, snells_2d(DxInCenter, n1, n2), label="Regular Snell's")
plt.plot(DxInCenter, smooth_snells(DxInCenter, stds, n1, n2), label=widthlabels)

plt.xlabel('Incoming direction x-component $(k_{x,in}/k_{in})$')
plt.ylabel('Outgoing direction x-component $(k_{x,out}/k_{out})$')
plt.grid()
plt.legend()
plt.title(f"Smooth Snell's with gaussian distribution\n$n_1={n1}, n_2={n2}$, {torch.get_default_dtype()}")
plt.show()


# Plot angle out vs angle in
theta_xincenter = torch.linspace(-np.pi/2, np.pi/2, N).reshape((-1, 1))

widthlabels = ['std=' + f'{w:.3f}' for w in stds.reshape(-1)]
plt.plot(theta_xincenter, np.arcsin(snells_2d(np.sin(theta_xincenter), n1, n2)), label="Regular Snell's")
plt.plot(theta_xincenter, np.arcsin(smooth_snells(np.sin(theta_xincenter), stds, n1, n2)), label=widthlabels)

plt.xlabel('$\\theta_{in}$ (rad)')
plt.ylabel('$\\theta_{out}$ (rad)')
plt.grid()
plt.xticks(np.array((-1/2, -1/4, 0, 1/4, 1/2))*np.pi,
    ('$-\\pi/2$', '$-\\pi/4$', '$0$', '$\\pi/4$', '$\\pi/2$'))
plt.yticks(np.array((-1/2, -1/4, 0, 1/4, 1/2))*np.pi,
    ('$-\\pi/2$', '$-\\pi/4$', '$0$', '$\\pi/4$', '$\\pi/2$'))
plt.legend()
plt.title("Smooth Snell's with gaussian distribution")
plt.show()


# Test 3D ray tracing version
origin, x, y, z = cartesian3d()
component_range = 4     # Controls max input angle = arctan(component_range)
Ny = 81
n_in = 1.3
n_out = 1.0
std1 = 0.08
std2 = 0.04
source_plane = CoordPlane(origin, x, component_range*y)
interface_plane = CoordPlane(origin + z, x, y)
end_plane = CoordPlane(origin + 2*z, x, y)

rays1 = []
rays1 += [point_source(source_plane, 1, Ny, refractive_index=n_in)]
rays1 += [rays1[-1].intersect_plane(interface_plane)]
rays1 += [snells_gaussian(rays1[-1], -z, std1, n_out=n_out)]
rays1 += [rays1[-1].propagate(1)]

rays2 = []
rays2 += [point_source(source_plane, 1, Ny, refractive_index=n_in)]
rays2 += [rays2[-1].intersect_plane(interface_plane)]
rays2 += [snells_gaussian(rays2[-1], -z, std2, n_out=n_out)]
rays2 += [rays2[-1].propagate(1)]

rays3 = []
rays3 += [point_source(source_plane, 1, Ny, refractive_index=n_in)]
rays3 += [rays3[-1].intersect_plane(interface_plane)]
rays3 += [snells(rays3[-1], -z, n_out=n_out)]
rays3 += [rays3[-1].propagate(1)]

# Plot 3D ray tracing version
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 7))
fig.dpi = 144

# Snell's with gaussian
plot_plane(ax[0], interface_plane, scale=component_range+1)
plot_rays(ax[0], rays1)
ax[0].set_aspect(1)
ax[0].set_title(f"Snell's with gaussian\n$n_1={n_in}, n_2={n_out:.0f}, \\sigma={std1}$")
ax[0].set_xlabel('optical axis (m)')
ax[0].set_ylabel('transverse axis (m)')

# Snell's with gaussian
plot_plane(ax[1], interface_plane, scale=component_range+1)
plot_rays(ax[1], rays2)
ax[1].set_aspect(1)
ax[1].set_title(f"Snell's with gaussian\n$n_1={n_in}, n_2={n_out:.0f}, \\sigma={std2}$")
ax[1].set_xlabel('optical axis (m)')
ax[1].set_ylabel('transverse axis (m)')

# Regular Snell's
plot_plane(ax[2], interface_plane, scale=component_range+1)
plot_rays(ax[2], rays3)
ax[2].set_aspect(1)
ax[2].set_title(f"Regular Snell's\n$n_1={n_in}, n_2={n_out}$")
ax[2].set_xlabel('optical axis (m)')
ax[2].set_ylabel('transverse axis (m)')
plt.show()

# Intensity and kx/k plot
intense_figure = plt.figure()
fig.dpi = 144
plt.plot(rays3[0].direction[:, :, 1].view(-1), rays3[-1].direction[:, :, 1].view(-1), label='$k_y/k$ out (Regular)')
plt.plot(rays1[0].direction[:, :, 1].view(-1), rays1[-1].direction[:, :, 1].view(-1), label=f'$k_y/k$ out, $\\sigma={std1:.2f}$')
plt.plot(rays2[0].direction[:, :, 1].view(-1), rays2[-1].direction[:, :, 1].view(-1), label=f'$k_y/k$ out, $\\sigma={std2:.2f}$')
plt.plot(rays1[0].direction[:, :, 1].view(-1), rays1[-1].intensity.view(-1), label=f'Intensity out, $\\sigma={std1:.2f}$')
plt.plot(rays2[0].direction[:, :, 1].view(-1), rays2[-1].intensity.view(-1), label=f'Intensity out, $\\sigma={std2:.2f}$')
plt.xlabel('Incident $k_y/k$')
plt.title(f"Raytracer Snell's law\n$n_1={n_in}, n_2={n_out}$")
plt.legend()
plt.grid()
plt.show()


# Test 3D ray tracing version softplus
rays_gauss = []
rays_gauss += [point_source(source_plane, 1, Ny, refractive_index=n_in)]
rays_gauss += [rays_gauss[-1].intersect_plane(interface_plane)]
rays_gauss += [snells_gaussian(rays_gauss[-1], -z, std1, n_out=n_out)]
rays_gauss += [rays_gauss[-1].propagate(1)]

rays_softp = []
rays_softp += [point_source(source_plane, 1, Ny, refractive_index=n_in)]
rays_softp += [rays_softp[-1].intersect_plane(interface_plane)]
rays_softp += [snells_softplus(rays_softp[-1], -z, std1, n_out=n_out)]
rays_softp += [rays_softp[-1].propagate(1)]

rays_snell = []
rays_snell += [point_source(source_plane, 1, Ny, refractive_index=n_in)]
rays_snell += [rays_snell[-1].intersect_plane(interface_plane)]
rays_snell += [snells(rays_snell[-1], -z, n_out=n_out)]
rays_snell += [rays_snell[-1].propagate(1)]

# Plot 3D ray tracing version
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 7))
fig.dpi = 144

# Snell's with gaussian
plot_plane(ax[0], interface_plane, scale=component_range+1)
plot_rays(ax[0], rays_gauss)
ax[0].set_aspect(1)
ax[0].set_title(f"Snell's with gaussian\n$n_1={n_in}, n_2={n_out:.0f}, \\sigma={std1}$")
ax[0].set_xlabel('optical axis (m)')
ax[0].set_ylabel('transverse axis (m)')

# Snell's with gaussian
plot_plane(ax[1], interface_plane, scale=component_range+1)
plot_rays(ax[1], rays_softp)
ax[1].set_aspect(1)
ax[1].set_title(f"Snell's with Softplus\n$n_1={n_in}, n_2={n_out:.0f}, \\sigma={std1}$")
ax[1].set_xlabel('optical axis (m)')
ax[1].set_ylabel('transverse axis (m)')

# Regular Snell's
plot_plane(ax[2], interface_plane, scale=component_range+1)
plot_rays(ax[2], rays_snell)
ax[2].set_aspect(1)
ax[2].set_title(f"Regular Snell's\n$n_1={n_in}, n_2={n_out}$")
ax[2].set_xlabel('optical axis (m)')
ax[2].set_ylabel('transverse axis (m)')
plt.show()

# Intensity and kx/k plot
intense_figure = plt.figure()
fig.dpi = 144
plt.plot(rays_snell[0].direction[:, :, 1].view(-1), rays_snell[-1].direction[:, :, 1].view(-1), label='$k_y/k$ out (Regular)')
plt.plot(rays_gauss[0].direction[:, :, 1].view(-1), rays_gauss[-1].direction[:, :, 1].view(-1), label=f'$k_y/k$ out (Gaussian), $\\sigma={std1:.3f}$')
plt.plot(rays_softp[0].direction[:, :, 1].view(-1), rays_softp[-1].direction[:, :, 1].view(-1), label=f'$k_y/k$ out (Softplus), $\\sigma={std1:.3f}$')
plt.plot(rays_gauss[0].direction[:, :, 1].view(-1), rays_gauss[-1].intensity.view(-1), label=f'Intensity out (Gaussian), $\\sigma={std1:.3f}$', linewidth=3, linestyle='dashed')
plt.plot(rays_softp[0].direction[:, :, 1].view(-1), rays_softp[-1].intensity.view(-1), label=f'Intensity out (Softplus), $\\sigma={std1:.3f}$')
plt.xlabel('Incident $k_y/k$')
plt.title(f"Raytracer Snell's law\n$n_1={n_in}, n_2={n_out}$, {torch.get_default_dtype()}")
plt.legend()
plt.grid()
plt.show()
