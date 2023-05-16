# Smooth Snell's law test
# Test a derived version of Snell's law based on an angle distribution, rather than a single angle

import torch
from torch import tensor
import numpy as np
import matplotlib.pyplot as plt

from vector_functions import ensure_tensor, cartesian3d
from optical import snells_gaussian, point_source, CoordPlane, snells
from plot_functions import plot_plane, plot_rays


torch.set_default_tensor_type('torch.DoubleTensor')


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
plt.title(f"Smooth Snell's with gaussian distribution\n{torch.get_default_dtype()}")
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
component_range = 4
Ny = 23
n_in = 1.3
n_out = 1.0
std = 0.08
source_plane = CoordPlane(origin, x, component_range*y)
interface_plane = CoordPlane(origin + z, x, y)
end_plane = CoordPlane(origin + 2*z, x, y)

rays1 = []
rays1 += [point_source(source_plane, 1, Ny, refractive_index=n_in)]
rays1 += [rays1[-1].intersect_plane(interface_plane)]
rays1 += [snells_gaussian(rays1[-1], -z, std, n_out=n_out)]
rays1 += [rays1[-1].propagate(1)]

rays2 = []
rays2 += [point_source(source_plane, 1, Ny, refractive_index=n_in)]
rays2 += [rays2[-1].intersect_plane(interface_plane)]
rays2 += [snells(rays2[-1], -z, n_out=n_out)]
rays2 += [rays2[-1].propagate(1)]

# Plot 3D ray tracing version
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
fig.dpi = 144
plot_plane(ax[0], interface_plane, scale=component_range+1)
plot_rays(ax[0], rays1)
ax[0].set_aspect(1)
ax[0].set_title(f"Snell's with gaussian\n$n_1={n_in}, n_2={n_out}, \\sigma={std}$")
ax[0].set_xlabel('optical axis (m)')
ax[0].set_ylabel('transverse axis (m)')

plot_plane(ax[1], interface_plane, scale=component_range+1)
plot_rays(ax[1], rays2)
ax[1].set_aspect(1)
ax[1].set_title(f"Regular Snell's\n$n_1={n_in}, n_2={n_out}$")
ax[1].set_xlabel('optical axis (m)')
ax[1].set_ylabel('transverse axis (m)')
plt.show()
