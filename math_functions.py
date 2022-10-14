"""
Mathematical functions.
"""

import torch


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
    sqrt_discr = torch.sqrt(discriminant)
    x1 = (-b - sqrt_discr) / (2*a)
    x2 = (-b + sqrt_discr) / (2*a)
    return (x1, x2)
