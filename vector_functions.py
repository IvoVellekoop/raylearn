"""Vector functions. Basic vector operations.

Terminology used:
('Mx...' denotes one or multiple dimensions of undefined length.)
Vector: A DxMx... torch tensor where the 0th dimension denotes vector components.
        Usually, D=3, representing vector dimensions x, y and z. Other dimensions are used for
        parallelization.
Scalar: A 1xMx... torch tensor where the 0th dimension has length 1. Or, in cases where all
        elements of the Tensor have the same value, this may be replaced by a Python float or
        double. Other dimensions are used for parallellization.
Note that for vector and scalar operations, the number of dimensions must match! Dimensionality of
e.g. 3x1x1 is not the same as 3 for torch tensors, and will yield incorrect results for the vector
operations!
"""

import torch


# Vector operations

def dot(v, w):
    """Dot product for DxMx... vectors, where D is spatial dimension."""
    return torch.sum(v*w, dim=0, keepdim=True)


def norm(v):
    """L2-norm for DxM... vectors, where D is spatial dimension."""
    return torch.norm(v, dim=0, keepdim=True)


def norm_square(v):
    """L2-norm squared for DxM... vectors, where D is spatial dimension."""
    return dot(v, v)


def unit(v):
    """Compute unit vectors for DxM... vectors, where D is spatial dimension."""
    return v / norm(v)


def projection(v, w):
    """Vector projection of vector v onto w for DxM... vectors, where D is spatial dimension."""
    wunit = unit(w)
    return dot(v, wunit) * wunit


def rejection(v, w):
    """Vector rejection of vector v onto w for DxM... vectors, where D is spatial dimension."""
    return v - projection(v, w)


def cross(v, w):
    """Cross product of vector v and w for 3xM... vectors, where 3 is spatial dimension."""
    ux = v[1]*w[2] - v[2]*w[1]
    uy = v[2]*w[0] - v[0]*w[2]
    uz = v[0]*w[1] - v[1]*w[0]
    return torch.stack((ux, uy, uz))


def rotate(v, u, theta):
    """Rotate vector v theta radians around unit vector u.
    For derivation see: http://ksuweb.kennesaw.edu/~plaval/math4490/rotgen.pdf"""
    if type(theta) == torch.Tensor:
        tensor_theta = theta
    else:
        tensor_theta = torch.tensor(theta)

    C = torch.cos(tensor_theta)
    S = torch.sin(tensor_theta)
    return (1-C)*dot(v, u)*u + C*v + S*cross(u, v)


# Vector factory functions
def vector(values, size):
    """Construct a single vector from given tuple and expand to size.

    The vector components can be passed as a Python tuple, from which a torch Tensor with the
    desired dimensions is constructed, as specified in the size argument. The 0th dimension of the
    torch Tensor represents the vector dimension. The higher dimensions are used for
    parallelization. In PyTorch, a Tensor of size (3) is not the same as a Tensor of size (3,1,1).
    This expansion of dimensions in PyTorch is only required for correctly applying operations.
    The values are copied by index; the data itself is not copied. No new memory is allocated.

    Example
    -------
        vector((1,2,3), (3,4,4)) returns a torch Tensor representing the vector (1,2,3)
        parallelized in a 4x4 array. The whole torch Tensor will have a size of (3,4,4).
    """
    viewsize = [size[0]] + ((len(size)-1) * [1])    # Construct viewsize   e.g. (3,1,1)
    return torch.Tensor(values).view(viewsize).expand(size)

### Define size once and then define factory functions for vector, vector array, scalar, scalar array?
### If I were to put the spatial dimension D as the trailing dimension, a size of (1,1,3) would be
### equivalent to (3). Which would mean the vector could have a default size of (3), which should
### match any case.
