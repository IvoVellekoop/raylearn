"""classes - Define classes Ray and Plane.

The main purpose of these classes is to bundle together relevant parameters like position,
direction and refractive index.

Terminology used:
('Mx...' denotes one or multiple dimensions of undefined length.)
Vector: A DxMx... torch tensor where the 0th dimension denotes vector components.
        Usually, D=3, representing vector dimensions x, y and z. Other dimensions are used for
        parallelization.
Scalar: A 1xMx... torch tensor where the first dimension has length 1. Or, in cases where all Ray
        elements have the same value, this may be replaced by a Python float or double. Other
        dimensions are used for parallellization.
Note that for vector and scalar operations, the number of dimensions must match! Dimensionality of
e.g. 3x1x1 is not the same as 3 for torch tensors, and will yield incorrect results for the vector
operations!
"""

import torch
from vector_functions import dot, cross, unit, norm_square
import copy


class Ray:
    """Ray class. Represents a (collection of) rays.

    A ray has a position in meters, a directional unit vector, and the
    refractive index of the medium. Optical element functions take a Ray as
    input and give a Ray as output. Note that the higher dimensions of Vector- and Scalar-Tensors
    can be used for parallelization.

    Attributes
    ----------
        position_m          Vector. Ray position in meters
        direction           Vector. Ray direction unit vector
        refractive_index    Scalar. Refractive index of medium
        pathlength          Scalar. Total optical pathlength in meters
        weight              Scalar. Total weight. Adjusts contribution to objective function.

    """

    def __init__(self, position_m, direction, refractive_index=1, pathlength_m=0, intensity=1, weight=1):
        self.position_m = position_m                # Vector array. Position in m
        self.direction = direction                  # Vector array. Direction unit vector
        self.refractive_index = refractive_index    # Scalar array. Refractive index of medium
        self.pathlength_m = pathlength_m            # Scalar array. Total optical pathlength in m
        self.intensity = intensity                  # Scalar array. Intensity of ray.
        self.weight = weight                        # Scalar array. Total weight in loss function.

    def intersect_plane(self, plane):
        """Return new Ray at intersection with Plane or CoordPlane."""
        distance_m = dot(plane.normal, plane.position_m - self.position_m) \
            / dot(plane.normal, self.direction)
        intersection = self.position_m + self.direction * distance_m
        return self.copy(position_m=intersection)

    def mask(self, mask_factors):
        """Mask Ray with given mask_factors array."""
        intensity = self.intensity * mask_factors
        return self.copy(intensity=intensity)

    def copy(self, **kwargs):
        """
        Create copy of this Ray, with some slight alterations passed as keyword arguments.
        Note that torch tensor attribute will reference the same value as the original Ray.
        """
        copiedray = copy.copy(self)
        copiedray.__dict__.update(**kwargs)
        return copiedray



class Plane:
    """Plane class.

    A plane has properties similar to a Ray (position and direction), but represents a plane
    perpendicular to its defined normal vector. It can be used to represent the position and
    orientation of for instance an ideal lens.

    Attributes
    ----------
        position_m          Vector. Plane position in meters
        normal              Vector. Unit vector of plane normal

    """

    def __init__(self, position_m, normal):
        self.position_m = position_m                # Vector array. Position in m
        self.normal = normal                        # Vector array. Plane normal as unit vector


class CoordPlane():
    """CoordPlane class.

    A coordinate plane is defined by a position vector and two component vectors x & y that define
    its coordinate system. It is suitable to represents for instance a camera plane. The normal
    attribute represents the unit vector perpendicular to the two component vectors, and is
    calculated on the fly when read. This makes the CoordPlane class mostly compatible with the
    Plane class. However, since the CoordPlane.normal attribute is calculated from the x & y
    component vectors on the fly, it is read only. The unit of the x & y component vectors depends
    on the application.

    Attributes
    ----------
        position_m          Vector. Plane position vector in meters.
        x                   Vector. x component vector.
        y                   Vector. y component vector.
        normal              Read only. Vector array. Unit vector of plane normal. When read,
                            this attribute is calculated on the fly.
    """

    def __init__(self, position_m, x, y):
        self.position_m = position_m
        self.x = x
        self.y = y

    @property
    def normal(self):
        """Compute normal vector of the plane. Computed from the x & y component vectors."""
        return unit(cross(self.x, self.y))

    def transform(self, rays):
        """Transform vector array to coordinates of the CoordPlane x & y."""
        p = rays.position_m - self.position_m
        x = dot(p, self.x) / norm_square(self.x)
        y = dot(p, self.y) / norm_square(self.y)
        return torch.cat((x, y))