"""
Mathematical functions.
"""

from vector_functions import ensure_tensor
from torch import Tensor, sqrt, isnan, nan, zeros_like, nn


def sqrt_zero(x_in, beta=5):
    """
    Square Root or Zero
    Returns the sqrt of x if x>0. Returns 0 otherwise. Works on torch Tensors of any shape.
    """
    x = ensure_tensor(x_in)
    # z = zeros_like(x)
    # mask = (x > 0)
    softplus = nn.Softplus(beta=beta)
    relu = nn.ReLU()
    # z[mask] = softplus(x).sqrt()[mask]
    z = relu(x).sqrt() + 1e-4
    # z = softplus(x).sqrt()
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
