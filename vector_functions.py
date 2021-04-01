"""Vector functions. Basic vector operations.

Terminology used:
('Mx...' denotes one or multiple dimensions of undefined length.)
Vector:   A DxMx... torch tensor where the first dimension denotes spatial components.
          Usually, D=3, representing spatial dimensions x, y and z. The other dimensions Mx... can
          be used as regular array dimensions.
Scalar:   A 1xMx... torch tensor where the first dimension has length 1. The other dimensions Mx...
          can be used as regular array dimensions.

"""

import torch


## Vector operations

def dot(v, w):
    """Dot product for DxMx... vectors, where D is spatial dimension."""
    return torch.sum(v*w, dim=0, keepdim=True)

def norm(v):
    """L2-norm for DxM... vectors, where D is spatial dimension."""
    return torch.norm(v, dim=0, keepdim=True)

def norm_square(v):
    """L2-norm squared for DxM... vectors, where D is spatial dimension."""
    return dot(v,v)

def unit(v):
    """Compute unit vectors for DxM... vectors, where D is spatial dimension."""
    return v / norm(v)

def projection(v, w):
    """Vector projection of vector v onto vector for DxM... vectors, where D is spatial dimension."""
    wunit = unit(w)
    return dot(v, wunit) * wunit

def rejection(v, w):
    """Vector rejection of vector v onto vector for DxM... vectors, where D is spatial dimension."""
    return v - projection(v, w)


## Vector factory functions
