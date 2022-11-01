"""
Plot functions.

Functions for plotting Rays, Planes, stuff to draw your defined optical elements, etc.
"""

import torch
from torch import stack, Tensor
import numpy as np
from vector_functions import norm, norm_square, unit, rejection, cartesian3d
from ray_plane import Plane, CoordPlane


def default_viewplane():
    """Define default viewplane for plot functions."""
    origin, x, y, z = cartesian3d()
    viewplane = CoordPlane(origin, z, y)
    return viewplane


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
    """
    Take list or tuple of rays and return tuple of ray positions. The positions are expanded
    according to broadcasting semantics, so the shapes are compatible (also for plotting).
    See also:
    https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics

    Input
    -----
        raylist     List or tuple of Ray objects.

    Output
    ------
        positions   Tuple of expanded position Tensors.
    """

    positions_tuple = tuple(ray.position_m for ray in raylist)      # Extract positions as tuple
    positions_expanded = torch.broadcast_tensors(*positions_tuple)  # Expand
    return positions_expanded


def plot_rays(ax, rays, viewplane=default_viewplane(), plotkwargs={}, fraction=1):
    """
    Plot rays

    Input
    -----
        ax          Matplotlib Axis object. Figure axis to plot on.
        rays        List or tuple of Ray objects. The ray positions will be plotted.
        viewplane   CoordPlane. Viewing plane to project positions onto.
        plotkwargs  Dictionary. Keyword arguments to be passed onto the plot function.
        fraction    Float. This fraction of randomly picked rays from the rays list will
                    be plotted. Can be useful when dealing with many rays, as this can be slow.
    """

    # Prepare variables
    positions = viewplane.transform_points(stack(ray_positions(rays))).unbind(-1)
    positions_hori = positions[0].view(len(rays), -1).detach().cpu()
    positions_vert = positions[1].view(len(rays), -1).detach().cpu()

    # Select a fraction of the rays
    randomgenerator = torch.random.manual_seed(1)
    mask = torch.rand((1, positions_hori.shape[-1]), generator=randomgenerator) < fraction
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
    return ln


def plot_plane(ax, plane_to_plot, scale=1, text1='', text2='', viewplane=default_viewplane(),
               plotkwargs={'color': 'black'}):
    """
    Plot a plane.

    A square orthogonal to the plane normal is projected onto the viewplane, resulting in a
    parallelogram or line. If plane_to_plot is a CoordPlane, its x- and y-vectors are used
    to define the square. If plane_to_plot is a Plane, orthonormal x- and y-vectors are created
    using a vector rejection of the viewplane on the plane_to_plot.

    Input
    -----
        ax              Matplotlib Axis object. Figure axis to plot on.
        plane_to_plot   Plane or CoordPlane. The Plane or CoordPlane that will be plotted.
        scale           Float. Scale factor for the projected square representing the plane.
        text1           String. Text label 1. Is put at one of the square corners.
        text2           String. Text label 2. Is put at another of the square corners.
        viewplane       CoordPlane. Viewing plane to project positions onto.
        plotkwargs      Dictionary. Keyword arguments to be passed onto the plot function.

    Output
    ------
        ln              Matplotlib Line2D object representing the plotted data.

    """
    # Define x and y vector lying on the CoordPlane or Plane to be plotted
    position_m = plane_to_plot.position_m.detach()

    if isinstance(plane_to_plot, CoordPlane):

        x = scale * plane_to_plot.x.detach()
        y = scale * plane_to_plot.y.detach()

    elif isinstance(plane_to_plot, Plane):
        # A Plane has no x- and y-vectors defined. So let's invent them.
        x_rej = rejection(viewplane.x, plane_to_plot.normal.detach())
        y_rej = rejection(viewplane.y, plane_to_plot.normal.detach())

        # Find suitable x-vector
        if norm(x_rej) > 0:
            x = scale * unit(x_rej).detach()
        else:
            x = Tensor((0., 0., 0.)).detach()

        # Find suitable y-vector
        if norm(y_rej) > 0:
            y = scale * unit(y_rej).detach()
        else:
            y = Tensor((0., 0., 0.)).detach()

    # Compute 4 corner points of plane square around plane position
    A = position_m + x + y
    B = position_m + x - y
    C = position_m - x - y
    D = position_m - x + y

    # Project points onto viewplane
    Ax, Ay = viewplane.transform_points(A).unbind(-1)
    Bx, By = viewplane.transform_points(B).unbind(-1)
    Cx, Cy = viewplane.transform_points(C).unbind(-1)
    Dx, Dy = viewplane.transform_points(D).unbind(-1)
    Px, Py = viewplane.transform_points(position_m).unbind(-1)
    arrow_scale = 0.1 * np.sqrt(norm_square(x) + norm_square(y))
    nx, ny = viewplane.transform_points(arrow_scale * plane_to_plot.normal.detach()).unbind(-1)

    parallelogram = ax.plot((Ax, Bx, Cx, Dx, Ax),
                            (Ay, By, Cy, Dy, Ay), **plotkwargs)
    arrow = ax.arrow(Px, Py, nx, ny, width=0.1*arrow_scale, **plotkwargs)
    ax.text(Bx, By, text1)
    ax.text(Dx, Dy, text2)
    return (parallelogram, arrow)


def plot_lens(ax, lensplane, f, scale=1, pretext1='', text2='', viewplane=default_viewplane(),
              plotkwargs={'color': 'black'}):
    """
    Plot a lens plane

    A square orthogonal to the lensplane normal is projected onto the viewplane, resulting in a
    parallelogram or line. See plot_plane for details.

    Input
    -----
        ax              Matplotlib Axis object. Figure axis to plot on.
        lensplane       Plane or CoordPlane. The Plane or CoordPlane that will be plotted.
        f               Float. Focal distance of lens for annotation.
        scale           Float. Scale factor for the projected square representing the plane.
        pretext         String. Extra pre-text for text label 1. Is put at one of the corners.
        viewplane       CoordPlane. Viewing plane to project positions onto.
        plotkwargs      Dictionary. Keyword arguments to be passed onto the plot function.

    Output
    ------
        ln              Matplotlib Line2D object representing the plotted data.

    """
    text1 = pretext1 + f' f={format_prefix(f, ".1f")}m'     # Write pretext and focal distance
    ln = plot_plane(ax, lensplane, scale, text1, text2, viewplane, plotkwargs)
    return ln
