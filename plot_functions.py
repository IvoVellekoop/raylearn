"""Plot functions."""

import torch
from torch import stack
import numpy as np
from vector_functions import unit, rejection
from ray_plane import Plane, CoordPlane


def format_prefix(number, formatspec='.2f'):
    """
    Converts a number to a string with a metric unit prefix, e.g. 2550 -> '2.55K'.
    Optional rounding.

    Input:
    number        Numeric. Value of number to convert to string.
    formatspec    Format specification string. Default: '.2f'.

    Output:       String of formatted value with unit prefix.

    See https://www.python.org/dev/peps/pep-0498/#format-specifiers
    for details on the formatspec.
    """

    # Detach torch Tensors
    if isinstance(number, torch.Tensor):
        value = number.detach().item()
    else:
        value = number

    # Special case 0
    if value == 0:
        return f'{value}'

    prefixes = ["a", "f", "p", "n", "µ", "m", "", "k", "M", "G", "T", "P", "E"]
    prefix_power = np.floor(np.log10(np.abs(value)) / 3)            # Power of 1000 of unit prefix
    prefix_index = int(prefix_power + 6)                            # Index of unit prefix
    value_in_prefix = value / 1000**prefix_power                    # Value in that unit prefix
    return f'{value_in_prefix:{formatspec}}{prefixes[prefix_index]}'


def ray_positions(raylist):
    """Take list or tuple of rays and return list of rays."""
    numels = tuple(torch.numel(x.position_m) for x in raylist)  # Total elements per Ray position
    biggest_raypos = raylist[np.argmax(numels)].position_m      # Ray with biggest position array

    # Expand each position array and return as tuple
    return tuple(x.position_m.expand_as(biggest_raypos) for x in raylist)


def plot_rays(ax, viewplane, rays, plotkwargs={}, fraction=1):
    """Plot rays."""

    ######## add default viewplane function

    # Prepare variables
    positions = viewplane.transform_points(stack(ray_positions(rays))).unbind(-1)
    positions_hori = positions[0].view(len(rays), -1).detach().cpu()
    positions_vert = positions[1].view(len(rays), -1).detach().cpu()

    # Select a fraction of the rays
    mask = torch.rand(1, positions_hori.shape[-1]) < fraction
    positions_hori_select = positions_hori.masked_select(mask).view(len(rays), -1)
    positions_vert_select = positions_vert.masked_select(mask).view(len(rays), -1)

    # Plot
    ln = ax.plot(positions_hori_select, positions_vert_select, '.-', **plotkwargs)
    return ln


def plot_coords(ax, coords, plotkwargs={'color': 'tab:blue'}):
    """Plot coordinates."""
    x, y = coords.unbind(-1)
    ln = ax.plot(x.detach().cpu().view(-1),
                 y.detach().cpu().view(-1), '.', **plotkwargs)
    ax.set_xlabel('x (pix)')
    ax.set_ylabel('y (pix)')
    return ln


def plot_plane(ax, viewplane, plane_to_plot, scale=1, text='', plotkwargs={'color': 'black'}):
    """Plot a plane.

    WIP: currently assumes: dimhori = 2, dimvert = 1, text at point B.
    Improvement: Choose a plotting plane
    """
    # Get properties
    position_m = plane_to_plot.position_m.detach()
    if isinstance(plane_to_plot, CoordPlane):
        x = scale * plane_to_plot.x.detach()
        y = scale * plane_to_plot.y.detach()
    elif isinstance(plane_to_plot, Plane):
        ###### Not a reliable way to get orthogonal basis vectors on the plane,
        ###### as this an cause /0 errors when vectors are parallel
        x = scale * unit(rejection(viewplane.normal, plane_to_plot.normal.detach()))
        y = scale * unit(rejection(viewplane.y, plane_to_plot.normal.detach()))

    # Compute 4 corner points of plane
    A = position_m + x + y
    B = position_m + x - y
    C = position_m - x - y
    D = position_m - x + y

    # Project points onto viewplane
    Ax, Ay = viewplane.transform_points(A).unbind(-1)
    Bx, By = viewplane.transform_points(B).unbind(-1)
    Cx, Cy = viewplane.transform_points(C).unbind(-1)
    Dx, Dy = viewplane.transform_points(D).unbind(-1)

    ln = ax.plot((Ax, Bx, Cx, Dx, Ax),
                 (Ay, By, Cy, Dy, Ay), **plotkwargs)
    ax.text(Bx, By, text)
    return ln


def plot_lens(ax, viewplane, lensplane, f, scale, pretext='', plotkwargs={'color': 'black'}):
    """Plot lens."""
    text = pretext + f' f={f*1e3:.2f}mm'
    ln = plot_plane(ax, viewplane, lensplane, scale, text, plotkwargs)
    return ln
