# Smooth Snell's law test
# Test a derived version of Snell's law based on an angle distribution, rather than a single angle

import torch
from torch import tensor
import numpy as np
import matplotlib.pyplot as plt

from vector_functions import ensure_tensor


# torch.set_default_tensor_type('torch.DoubleTensor')


def snells(Dxin, n1, n2):
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
    # DxInCrit = ensure_tensor(n2/n1)
    # intensity = (torch.erf((DxInCrit - DxInCenter) / (np.sqrt(2)*width)) +
    #              torch.erf((DxInCrit + DxInCenter) / (np.sqrt(2)*width))) / 2
    # DxInMean = DxInCenter - ((-1 + torch.exp((2*DxInCrit*DxInCenter) / width**2)) * np.sqrt(2/np.pi) * width) / (2*intensity * torch.exp((DxInCrit + DxInCenter)**2 / (2*width**2)))
    # DxOutMean = DxInMean * n1/n2
    # return DxOutMean

    DxInCrit = ensure_tensor(n2/n1)
    intensity = (torch.erf((DxInCrit + DxInCenter) / (np.sqrt(2)*std)) +
                 torch.erf((DxInCrit - DxInCenter) / (np.sqrt(2)*std))) / 2
    DxInMean = DxInCenter + np.sqrt(2/np.pi) * std * (\
          torch.exp(
            -((DxInCenter + DxInCrit)**2/(2*std**2)) \
            - torch.log(2*intensity))
        - torch.exp(
            (2*DxInCenter*DxInCrit)/std**2 \
            - (DxInCenter + DxInCrit)**2/(2*std**2) \
            - torch.log(2*intensity)))
    DxOutMean = DxInMean * n1/n2
    return DxOutMean

    # Check integration limits for n1<n2
    # Compute DzOut
    # Also apply intensity factor to ray
    # Fresnel equations
    # Add check for extreme exponents with poor precision


N = 800
DxInCenter = torch.linspace(-1, 1, N).view((-1, 1))

# width = tensor((0.2, 0.1, 0.05, 0.02, 0.01)).view((1, -1))
stds = tensor((0.0892, 0.04,)).view((1, -1))
n1 = 1.5
n2 = 1.3

# Plot DxOut vs DxIn
widthlabels = ['std=' + f'{w:.3f}' for w in stds.view(-1)]
plt.plot(DxInCenter, snells(DxInCenter, n1, n2), label="Regular Snell's")
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
plt.plot(theta_xincenter, np.arcsin(snells(np.sin(theta_xincenter), n1, n2)), label="Regular Snell's")
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
