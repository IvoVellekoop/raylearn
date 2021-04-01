"""classes - Define classes Ray and Plane.

The main purpose of these classes is to bundle together relevant parameters like position,
direction and refractive index.

Terminology used:
('Mx...' denotes one or multiple dimensions of undefined length.)
Vector: A DxMx... torch tensor where the first dimension denotes spatial components.
        Usually, D=3, representing spatial dimensions x, y and z.
Scalar: A 1xMx... torch tensor where the first dimension has length 1. Or, in cases where all Ray
        elements have the same value, this may be replaced by a Python float or double.

"""

from vector_functions import dot


class Ray:
    """Ray.

    A ray has a position in meters, a directional unit vector, and the
    refractive index of the medium. Optical element functions take a Ray as
    input and give a Ray as output.

    Parameters
    ----------
        position_m          Vector array. Ray position in meters
        direction           Vector array. Ray direction unit vector
        refractive_index    Scalar array. Refractive index of medium
        pathlength          Scalar array. Total optical pathlength in meters
        weight              Scalar array. Total weight. Adjusts contribution to objective function.

    """

    def __init__(self, position_m, direction, refractive_index=1, pathlength_m=0, weight=1):
        self.position_m = position_m                # Vector array. Position in m
        self.direction = direction                  # Vector array. Direction unit vector
        self.refractive_index = refractive_index    # Scalar array. Refractive index of medium
        self.pathlength = pathlength_m              # Scalar array. Total optical pathlength in m
        self.weight = weight                        # Scalar array. Total weight in loss function.


    def intersect_plane(self, plane):
        """Vector point of ray intersection with plane. Return Vector array position in meters."""
        distance_m = dot(plane.normal, plane.position_m - self.position_m) \
                 / dot(plane.normal, self.direction)
        return self.position_m + self.direction * distance_m


class Plane:
    """Plane.

    A plane has properties similar to a Ray (position and direction), but represents a plane
    perpendicular to its defined normal vector. It can be used to represent the position and
    orientation of for instance an ideal lens.

    Parameters
    ----------
        position_m          Vector array. Plane position in meters
        normal              Vector array. Unit vector of plane normal

    """

    def __init__(self, position_m, normal):
        self.position_m = position_m                # Vector array. Position in m
        self.normal = normal                        # Vector array. Plane normal as unit vector