"""Plot functions."""

import torch
from torch import tensor, stack
import numpy as np
from vector_functions import unit, cross, rejection
from ray_plane import Plane, CoordPlane


def ray_positions(raylist, fraction=1):
    """Take list or tuple of rays and return list of rays."""
    return tuple(x.position_m for x in raylist)


def plot_rays(ax, rays, plotkwargs={}, fraction=1):
    """Plot rays.

    WIP: currently assumes: dimhori = 2, dimvert = 1, text at point B.
    Improvement: Choose a plotting plane
    """
    # Prepare variables
    dimhori = 2
    dimvert = 1
    dimlabels = ['x (m)', 'y (m)', 'z (m)']
    positions = stack(ray_positions(rays, 0.1)).unbind(-1)
    positions_hori = positions[dimhori].view(len(rays), -1).detach().cpu()
    positions_vert = positions[dimvert].view(len(rays), -1).detach().cpu()

    # Select a fraction of the rays
    mask = torch.rand(1, positions_hori.shape[-1]) < fraction
    positions_hori_select = positions_hori.masked_select(mask).view(len(rays), -1)
    positions_vert_select = positions_vert.masked_select(mask).view(len(rays), -1)
#    plot_plane = CoordPlane(tensor((0.,0.,0.)), tensor((0.,0.,1.)), tensor((0.,1.,0.)))

    # Plot
    ln = ax.plot(positions_hori_select, positions_vert_select, '.-', **plotkwargs)
    ax.axis('equal')
    ax.set_xlabel(dimlabels[dimhori])
    ax.set_ylabel(dimlabels[dimvert])
    return ln


def plot_coords(ax, coords, plotkwargs={'color': 'tab:blue'}):
    """Plot coordinates."""
    x, y = coords.unbind(-1)
    ln = ax.plot(x.detach().cpu().view(-1),
                 y.detach().cpu().view(-1), '.', **plotkwargs)
    ax.axis('equal')
    ax.set_xlabel('x (pix)')
    ax.set_ylabel('y (pix)')
    return ln


def plot_plane(ax, plane, scale=1, text='', plotkwargs={'color': 'black'}):
    """Plot a plane.

    WIP: currently assumes: dimhori = 2, dimvert = 1, text at point B.
    Improvement: Choose a plotting plane
    """
    # Get properties
    position_m = plane.position_m.detach()
    if isinstance(plane, CoordPlane):
        x = scale * plane.x.detach()
        y = scale * plane.y.detach()
    elif isinstance(plane, Plane):
        x = scale * unit(rejection(tensor((1., 0., 0.)), plane.normal.detach()))
        y = scale * unit(rejection(tensor((0., 1., 0.)), plane.normal.detach()))

    # Compute 4 corner points of plane
    A = position_m + x + y
    B = position_m + x - y
    C = position_m - x - y
    D = position_m - x + y

    Ax, Ay, Az = A.unbind(-1)
    Bx, By, Bz = B.unbind(-1)
    Cx, Cy, Cz = C.unbind(-1)
    Dx, Dy, Dz = D.unbind(-1)

    ln = ax.plot((Az, Bz, Cz, Dz, Az),
                 (Ay, By, Cy, Dy, Ay), **plotkwargs)
    ax.text(Bz, By, text)
    return ln


def plot_lens(ax, plane, f, scale, pretext='', plotkwargs={'color': 'black'}):
    """Plot lens."""
    x_global = tensor((1., 0., 0.))
    x_plane = unit(rejection(x_global, plane.normal))
    y_plane = cross(plane.normal, x_plane)

    tilt = np.arcsin(float(plane.normal.unbind(-1)[1].detach())) * 180/np.pi
    text = pretext + f' f={f*1e3:.2f}mm'
    coordplane = CoordPlane(plane.position_m, x_plane, y_plane)
    ln = plot_plane(ax, coordplane, scale, text, plotkwargs)
    return ln
