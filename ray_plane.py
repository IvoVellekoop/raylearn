"""classes - Define classes Ray and Plane.

The main purpose of these classes is to bundle together relevant parameters like position,
direction and refractive index.

For clarification of the meaning in this context of terminology like vectors and scalars,
see vector_functions.py.
"""

import torch
from vector_functions import dot, cross, unit, norm_square, nancheck
from testing import comparetensors, checkunitvector
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
        pathlength_m        Scalar. Total optical pathlength in meters. When performing operations,
                            the propagation sign (i.e. increasing or decreasing pathlength) is
                            automatically determined from direction and position.
        intensity           Scalar. Intensity of the light ray.
        weight              Scalar. Total weight. Adjusts contribution to objective function.

    """

    @nancheck
    def __init__(self, position_m, direction, refractive_index=1, pathlength_m=0, intensity=1, weight=1, do_plot=1):
        self.position_m = position_m                # Vector. Position in m
        assert checkunitvector(direction)
        self.direction = direction                  # Vector. Direction unit vector
        self.refractive_index = refractive_index    # Scalar. Refractive index of medium
        self.pathlength_m = pathlength_m            # Scalar. Total optical pathlength in m
        self.intensity = intensity                  # Scalar. Intensity of ray.
        self.weight = weight                        # Scalar. Total weight in loss function.
        self.do_plot = do_plot                      # Scalar. Toggle to plot specific rays or not.

    @nancheck
    def propagate(self, distance_m):
        """Propagate ray a defined distance and return as new Ray."""
        destination = self.position_m + self.direction * distance_m
        new_pathlength_m = self.pathlength_m + self.refractive_index * distance_m
        return copy_update(self, position_m=destination, pathlength_m=new_pathlength_m)

    def intersect_plane(self, plane):
        """Return new Ray at intersection with Plane or CoordPlane."""
        distance_m = self.ray_distance_to_plane(plane)
        return self.propagate(distance_m)

    def ray_distance_to_plane(self, plane):
        """Return signed distance to Plane along Ray."""
        return dot(plane.normal, plane.position_m - self.position_m) \
            / dot(plane.normal, self.direction)

    def mask(self, mask_factors):
        """Mask Ray with given mask_factors array."""
        intensity = self.intensity * mask_factors
        return copy_update(self, intensity=intensity)


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

    @nancheck
    def __init__(self, position_m, normal):
        self.position_m = position_m                # Vector array. Position in m
        self.normal = normal                        # Vector array. Plane normal as unit vector
        assert checkunitvector(normal)


class CoordPlane():
    """CoordPlane class.

    A coordinate plane is defined by a position vector and two component vectors x & y that define
    its coordinate system. It is suitable to represent for instance a camera plane. The normal
    attribute represents the unit vector perpendicular to the two component vectors, and is
    calculated on the fly when read. This makes the CoordPlane class mostly compatible with the
    Plane class. The unit of the x & y component vectors depends on the application.

    Attributes
    ----------
        position_m          Vector. Plane position vector in meters.
        x                   Vector. x component vector.
        y                   Vector. y component vector.
        normal              Read only. Vector array. Unit vector of plane normal. When read,
                            this attribute is calculated on the fly.
    """

    @nancheck
    def __init__(self, position_m, x, y):
        self.position_m = position_m
        self.x = x
        self.y = y
        assert comparetensors(dot(x, y), 0)

    @property
    def normal(self):
        """Compute normal vector of the plane. Computed from the x & y component vectors."""
        return unit(cross(self.x, self.y))

    @nancheck
    def transform_points(self, points_m):
        """Transform vector array to coordinates of the CoordPlane x & y."""
        p = points_m - self.position_m
        x = dot(p, self.x) / norm_square(self.x)
        y = dot(p, self.y) / norm_square(self.y)
        return torch.cat((x, y), -1)

    def transform_rays(self, rays):
        """Transform Ray position to coordinates of the CoordPlane x & y."""
        ##### Away with ya!
        return self.transform_points(rays.position_m)

    @nancheck
    def transform_direction(self, direction):
        """
        Transform vector array to coordinates of the CoordPlane x & y
        (no offset, suitable for direction to extract sine of angle).
        """
        x = dot(direction, self.x) / norm_square(self.x)
        y = dot(direction, self.y) / norm_square(self.y)
        return torch.cat((x, y), -1)
    

@nancheck
def translate(instance, v):
    """
    Translate object with position_m property. Returns a new instance.

    Input
    -----
        instance    Object that has a position_m property.
        v           Relative translation vector.

    Output
    ------
        instance    New instance of the object class, with new position_m.
    """
    new_position_m = instance.position_m + v
    copied_object = copy.copy(instance)
    copied_object.position_m = new_position_m
    return copied_object


def copy_update(instance, **kwargs):
    """
    Create copy of this object, with some slight alterations passed as keyword arguments.
    Note that torch tensor attributes will reference the same value as the original object.
    """
    copied_object = copy.copy(instance)
    copied_object.__dict__.update(**kwargs)
    return copied_object


def detach(instance):
    """
    Create copy of this object, with all Tensor properties detached from the computational graph.
    """
    dict_detached = {}
    for kw, arg in instance.__dict__.items():           # Loop over keyword arguments
        if isinstance(arg, torch.Tensor):               # Only check torch Tensors
            dict_detached[kw] = arg.detach()            # Add detached copy of each Tensor property

    return copy_update(instance, **dict_detached)


# Custom types
PlaneType = Plane | CoordPlane
