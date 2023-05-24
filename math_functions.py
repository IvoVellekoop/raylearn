"""
Mathematical functions.
"""

from vector_functions import ensure_tensor, components
from torch import Tensor, sqrt, isnan, nan, zeros_like, nn, cat, where


def sqrt_zero(x_in, epsilon=1e-4):
    """
    Square Root or Zero
    Returns the sqrt of x if x>0. Returns 0 otherwise. Works on torch Tensors of any shape.
    """
    x = ensure_tensor(x_in)
    relu = nn.ReLU()
    z = relu(x).sqrt() + epsilon
    return z


def solve_quadratic(a, b, c):
    """
    Solve Quadratic
    Solve the real quadratic equation ax^2 + bx + c = 0 for x.

    Input
    ------
    a, b, c     Coefficients of the quadratic equation

    Output
    ------
    xsols       Tuple containing the solutions to the equation. If the equation has no real
                solutions, a tuple with torch nans is returned.
    """

    discriminant = b*b - 4*a*c
    sqrt_discr = sqrt(discriminant)

    if sqrt_discr.isnan().any():
        pass

    x1 = (-b - sqrt_discr) / (2*a)
    x2 = (-b + sqrt_discr) / (2*a)
    return (x1, x2)


def sign(t: Tensor) -> Tensor:
    """Sign function of torch tensor that retains NaNs."""
    t_sign = t.sign()
    t_sign[isnan(t)] = nan
    return t_sign


def pyramid(size: Tensor, coords: Tensor) -> Tensor:
    """
    Compute abs x,y distance to nearest border for points inside rectangle, box, etc. Points outside
    will return 0.

    Input
    -----
        size        Size (e.g. width and height) of the rectangle. Shape: ...xD
        coords      Coordinates. Shape: ...xD

    Output
    ------
        Absolute distance to nearest border for points inside. Points outside will return 0.
    """

    x, y = components(coords)
    xdist, ydist = components((size/2 - coords.abs()).relu())
    return where(xdist < ydist,
                 cat((xdist, zeros_like(ydist)), dim=-1),
                 cat((zeros_like(xdist), ydist), dim=-1))
