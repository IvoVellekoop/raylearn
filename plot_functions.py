"""Plot functions."""

import torch
import numpy as np
from vector_functions import unit, cross, rejection, vector
from ray_plane import CoordPlane


def ray_positions(raylist):
    """Take list or tuple of rays and return list of rays."""
    return tuple(x.position_m for x in raylist)


def plot_rays(ax, positions, plotkwargs={}):
    """Plot rays."""
    # Plot
    dimhori = 2
    dimvert = 1

    # Prepare variables
    dimlabels = ['x (m)', 'y (m)', 'z (m)']
    positions_hori = torch.stack(positions)[:, dimhori, :, :].view(
        len(positions), -1).detach().cpu()
    positions_vert = torch.stack(positions)[:, dimvert, :, :].view(
        len(positions), -1).detach().cpu()

    # Plot axure
    ln = ax.plot(positions_hori, positions_vert, '.-', **plotkwargs)
    ax.axis('equal')
    ax.set_xlabel(dimlabels[dimhori])
    ax.set_ylabel(dimlabels[dimvert])
    return ln


def plot_coords(ax, coords, plotkwargs={'color': 'tab:blue'}):
    """Plot coordinates."""
    ln = ax.plot(coords[0].detach().cpu().view(-1),
            coords[1].detach().cpu().view(-1), '.', **plotkwargs)
    ax.axis('equal')
    ax.set_xlabel('x (pix)')
    ax.set_ylabel('y (pix)')
    return ln


def plot_plane(ax, coordplane, text='', plotkwargs={'color': 'black'}):
    """Plot a plane.

    WIP: currently assumes: dimhori = 2, dimvert = 1, text at point B.
    """
    # Get properties
    position_m = coordplane.position_m
    x = coordplane.x
    y = coordplane.y

    # Compute 4 corner points of plane
    A = position_m + x + y
    B = position_m + x - y
    C = position_m - x - y
    D = position_m - x + y

    ln = ax.plot((A[2], B[2], C[2], D[2], A[2]),
                 (A[1], B[1], C[1], D[1], A[1]), **plotkwargs)
    ax.text(B[2], B[1], text)
    return ln


def plot_lens(ax, plane, f, width, plotkwargs={'color': 'black'}):
    """Plot lens."""
    shape = plane.normal.shape
    x_global = vector((1, 0, 0), shape)
    x_plane = width * unit(rejection(x_global, plane.normal))
    y_plane = cross(plane.normal, x_plane)

    tilt = np.arcsin(float(plane.normal[1].detach())) * 180/np.pi
    text = f' f={f*1e3:.1f}mm\n tilt={tilt:.2f}$\degree$'
    coordplane = CoordPlane(plane.position_m, x_plane, y_plane)
    ln = plot_plane(ax, coordplane, text, plotkwargs)
    return ln
