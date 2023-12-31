"""Vector functions. Basic vector operations.

Terminology used:
('...xM' denotes one or multiple dimensions of undefined length.)
Vector: A ...xMxD torch tensor where the last dimension denotes vector components.
        Usually, D=3, representing vector dimensions x, y and z. Other dimensions are used for
        parallelization. Vector Tensors can be broken up into their vector components by using
        the unbind method, and can be put together as a vector Tensor again using stack.
        Examples:
        vx, vy, vz = v.unbind(-1)       # Removes the vector dimension
        vx, vy, vz = components(v)      # Keeps the vector dimension by doing unsqueeze(-2) first
        w = torch.stack((wx, wy, wz), dim=-1)
Scalar: A ...xMx1 torch tensor where the last dimension has length 1. Or, in cases where all
        elements of the Tensor have the same value, this may be replaced by a Python float or
        double. Other dimensions are used for parallellization. (Although note that a few PyTorch
        Tensor functions, like torch.cos, will not accept native Python floats.)
Parallelization can be used for many things. For instance in a Ray object that represents rays of
different positions and/or directions, or Plane or CoordPlane objects with different positions or
orientations, e.g. for a galvo mirror plane.

For complete broadcasting semantics, see https://pytorch.org/docs/stable/notes/broadcasting.html
"""

import torch
from torch import Tensor
from typing import Sequence
from functools import wraps


# Custom types
Scalar = float | Tensor


def nancheck(func):
    """
    Decorator that checks all torch Tensor inputs for NaNs and Infs,
    and raises a ValueError if one is found.
    """
    def check_arg(arg):
        if isinstance(arg, torch.Tensor):               # Only check torch Tensors
            if arg.isnan().any():
                # raise ValueError('NaN detected!')
                pass
            if arg.isinf().any():
                # raise ValueError('Infinity detected!')
                pass

    @wraps(func)
    def nanchecked_function(*args, **kwargs):
        for arg in args:                                # Loop over positional arguments
            check_arg(arg)
        for kw, arg in kwargs.items():                  # Loop over keyword arguments
            check_arg(arg)
        return func(*args, **kwargs)

    return nanchecked_function


def ensure_tensor(x):
    """
    Convert input to a torch Tensor if it's not a torch Tensor.
    """
    if isinstance(x, Tensor):
        return x
    else:
        return torch.tensor(x)


# Vector operations

@nancheck
def dot(v: Tensor, w: Tensor) -> Tensor:
    """Dot product for ...xMxD vectors, where D is vector dimension."""
    return torch.sum(v*w, dim=-1, keepdim=True)


@nancheck
def norm(v: Tensor) -> torch.Tensor:
    """L2-norm for ...xMxD vectors, where D is vector dimension."""
    return torch.norm(v, dim=-1, keepdim=True)


@nancheck
def norm_square(v: Tensor) -> Tensor:
    """L2-norm squared for ...xMxD vectors, where D is vector dimension."""
    return dot(v, v)


@nancheck
def unit(v: Tensor) -> Tensor:
    """Compute unit vectors for ...xMxD vectors, where D is vector dimension."""
    if (norm(v) == 0).any():
        pass
    return v / norm(v)


@nancheck
def projection(v: Tensor, w: Tensor) -> Tensor:
    """Vector projection of vector v onto w for ...xMxD vectors, where D is vector dimension."""
    wunit = unit(w)
    return dot(v, wunit) * wunit


@nancheck
def rejection(v: Tensor, w: Tensor) -> Tensor:
    """Vector rejection of vector v onto w for ...xMxD vectors, where D is vector dimension."""
    return v - projection(v, w)


@nancheck
def reflection(v: Tensor, w: Tensor) -> Tensor:
    """Reflect vector v along vector w for ...xMxD vectors, where D is vector dimension."""
    return v - 2 * projection(v, w)


@nancheck
def cross(v: Tensor, w: Tensor) -> Tensor:
    """Cross product of vector v and w for ...xMx3 vectors, where 3 is vector dimension."""
    vx, vy, vz = v.unbind(-1)
    wx, wy, wz = w.unbind(-1)
    ux = vy*wz - vz*wy
    uy = vz*wx - vx*wz
    uz = vx*wy - vy*wx
    return torch.stack((ux, uy, uz), dim=-1)


@nancheck
def rotate(v: Tensor, u: Tensor, theta: Scalar) -> Tensor:
    """
    Rotate vector v theta radians around unit vector u, using Rodriques' rotation formula.

    For derivation see:
    - http://ksuweb.kennesaw.edu/~plaval/math4490/rotgen.pdf
    - https://en.wikipedia.org/w/index.php?title=Rodrigues%27_rotation_formula
    """
    tensor_theta = ensure_tensor(theta)

    C = torch.cos(tensor_theta)
    S = torch.sin(tensor_theta)
    return (1-C)*dot(v, u)*u + C*v + S*cross(u, v)


@nancheck
def area_para(v: Tensor, w: Tensor) -> Tensor:
    """Compute parallelogram area for ...xMx2 vectors, where 2 is vector dimension."""
    vx, vy = components(v)
    wx, wy = components(w)
    return vx*wy - vy*wx


@nancheck
def components(v: Tensor) -> Sequence[Tensor]:
    """
    Components
    Return spatial vector components vx,vy,vz... of ...xMxD vector v as tuple of ...xMx1 scalars.

    Example:
    vx, vy, vz = components(v)
    """
    return v.unsqueeze(-2).unbind(-1)


def cartesian3d() -> Sequence[Tensor]:
    """Return default tuple of an origin vector and 3 orthonormal 3D vectors."""
    origin = torch.tensor((0., 0., 0.))
    x = torch.tensor((1., 0., 0.))
    y = torch.tensor((0., 1., 0.))
    z = torch.tensor((0., 0., 1.))
    return origin, x, y, z
