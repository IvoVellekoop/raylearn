"""
Mathematical functions.
"""

from torch import Tensor, sqrt, isnan, nan


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
