"""Vector functions. Basic vector operations.

Terminology used:
('...xM' denotes one or multiple dimensions of undefined length.)
Vector: A ...xMxD torch tensor where the last dimension denotes vector components.
        Usually, D=3, representing vector dimensions x, y and z. Other dimensions are used for
        parallelization. Vector Tensors can be broken up into their vector components by using
        the unbind method, and can be put together as a vector Tensor again using stack.
        Example:
        vx, vy, vz = v.unbind(-1)
        w = torch.stack((wx, wy, wz), dim=-1)
Scalar: A ...xMx1 torch tensor where the last dimension has length 1. Or, in cases where all
        elements of the Tensor have the same value, this may be replaced by a Python float or
        double. Other dimensions are used for parallellization. (Although note that a few PyTorch
        Tensor functions, like torch.cos, will not accept native Python floats.)

For complete broadcasting semantics, see https://pytorch.org/docs/stable/notes/broadcasting.html
"""

import torch


# Vector operations

def dot(v, w):
    """Dot product for ...xMxD vectors, where D is vector dimension."""
    return torch.sum(v*w, dim=-1, keepdim=True)


def norm(v):
    """L2-norm for ...xMxD vectors, where D is vector dimension."""
    return torch.norm(v, dim=-1, keepdim=True)


def norm_square(v):
    """L2-norm squared for ...xMxD vectors, where D is vector dimension."""
    return dot(v, v)


def unit(v):
    """Compute unit vectors for ...xMxD vectors, where D is vector dimension."""
    return v / norm(v)


def projection(v, w):
    """Vector projection of vector v onto w for ...xMxD vectors, where D is vector dimension."""
    wunit = unit(w)
    return dot(v, wunit) * wunit


def rejection(v, w):
    """Vector rejection of vector v onto w for ...xMxD vectors, where D is vector dimension."""
    return v - projection(v, w)


def cross(v, w):
    """Cross product of vector v and w for ...xMx3 vectors, where 3 is vector dimension."""
    vx, vy, vz = v.unbind(-1)
    wx, wy, wz = w.unbind(-1)
    ux = vy*wz - vz*wy
    uy = vz*wx - vx*wz
    uz = vx*wy - vy*wx
    return torch.stack((ux, uy, uz), dim=-1)


def rotate(v, u, theta):
    """
    Rotate vector v theta radians around unit vector u.

    For derivation see: http://ksuweb.kennesaw.edu/~plaval/math4490/rotgen.pdf
    """
    if type(theta) == torch.Tensor:
        tensor_theta = theta                # Already a Tensor
    else:
        tensor_theta = torch.Tensor((theta,))  # Turn into Tensor

    C = torch.cos(tensor_theta)
    S = torch.sin(tensor_theta)
    return (1-C)*dot(v, u)*u + C*v + S*cross(u, v)
