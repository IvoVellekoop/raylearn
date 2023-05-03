# Smooth Snell's law test
# Test a derived version of Snell's law based on an angle distribution, rather than a single angle

import numpy as np
import matplotlib.pyplot as plt


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
        total_weight = (arccot(n1*width / (n2-n1*DxInCenter))
                      + arccot(n1*width / (n2+n1*DxInCenter))) / np.pi
        DxOutMean = \
            n1/n2 * (DxInCenter + width/(2*np.pi * total_weight) * (
                np.log((n2-n1*DxInCenter)**2 / n1**2 + width**2)
              - np.log((n2+n1*DxInCenter)**2 / n1**2 + width**2)))
        return DxOutMean


N = 800
Dxincenter = np.linspace(-1, 1, N).reshape((-1, 1))

width = np.array((0.2, 0.1, 0.04, 0.01)).reshape((1, -1))
n1 = 1.5
n2 = 1.3

# Plot DxOut vs DxIn
widthlabels = ['Width=' + str(w) for w in width.reshape(-1)]
plt.plot(Dxincenter, smooth_snells_lorentzian_2d(Dxincenter, width, n1, n2), label=widthlabels)
plt.plot(Dxincenter, snells(Dxincenter, n1, n2), label="Regular Snell's")

plt.xlabel('$\\sin(\\theta_{in})$')
plt.ylabel('$\\sin(\\theta_{out})$')
plt.grid()
plt.legend()
plt.title("Snell's law Lorentzian $\\sin(\\theta)$ distribution")
plt.show()


# Plot angle out vs angle in
theta_xincenter = np.linspace(-np.pi/2, np.pi/2, N).reshape((-1, 1))

widthlabels = ['Width=' + str(w) for w in width.reshape(-1)]
plt.plot(theta_xincenter, np.arcsin(smooth_snells_lorentzian_2d(np.sin(theta_xincenter), width, n1, n2)), label=widthlabels)
plt.plot(theta_xincenter, np.arcsin(snells(np.sin(theta_xincenter), n1, n2)), label="Regular Snell's")

plt.xlabel('$\\theta_{in}$ (rad)')
plt.ylabel('$\\theta_{out}$ (rad)')
plt.grid()
plt.xticks(np.array((-1/2, -1/4, 0, 1/4, 1/2))*np.pi,
    ('$-\\pi/2$', '$-\\pi/4$', '$0$', '$\\pi/4$', '$\\pi/2$'))
plt.yticks(np.array((-1/2, -1/4, 0, 1/4, 1/2))*np.pi,
    ('$-\\pi/2$', '$-\\pi/4$', '$0$', '$\\pi/4$', '$\\pi/2$'))
plt.legend()
plt.title("Snell's law Lorentzian $\\sin(\\theta)$ distribution")
plt.show()
