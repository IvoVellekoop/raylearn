"""Optical functions.

Functions that represent optical elements. They have Ray objects as input and/or output.

For clarification of the meaning in this context of terminology like vectors and scalars,
see vector_functions.py.
"""

import torch
import numpy as np

from testing import checkunitvector
from vector_functions import unit, dot, norm_square, rejection, reflection, rotate, components
from ray_plane import Ray, Plane


def collimated_source(sourceplane, Nx, Ny, **raykwargs):
    """Collimated source.
    Return a Ray object with Nx by Ny by D position_m, where D is vector dimension (usually 3).
    Position and grid spacing determined by input sourceplane and Nx & Ny.

    Input
    -----
        sourceplane     CoordPlane. Position_m defines the center position. The x and y vectors
                        define the span of the rays: distance in meter from center to outer ray.
        Nx              Number of ray elements along x plane direction.
        Ny              Number of ray elements along y plane direction.
        raykwargs       Additional properties to pass to the Ray object.

    Output
    ------
    A Ray object of collimated rays with Nx by Ny by 3 position_m. Direction is defined by
    sourceplane normal vector.
    """
    x_array = sourceplane.x * torch.linspace(-1.0, 1.0, Nx).view(Nx, 1, 1)
    y_array = sourceplane.y * torch.linspace(-1.0, 1.0, Ny).view(1, Ny, 1)
    position = sourceplane.position_m + x_array + y_array
    return Ray(position, sourceplane.normal, **raykwargs)


def point_source(sourceplane, Nx, Ny, **raykwargs):
    """Point source with limited opening angle.
    Return a Ray object with Nx by Ny by D direction, where D is vector dimension (usually 3).
    Position and grid spacing determined by input sourceplane and Nx & Ny.

    Input
    -----
        sourceplane     CoordPlane. Position_m defines the center position. The x and y vectors
                        define the aperture of the rays: tan(angle) between center and outer ray.
        Nx              Number of ray elements along x plane direction.
        Ny              Number of ray elements along y plane direction.
        raykwargs       Additional properties to pass to the Ray object.

    Output
    ------
    A Ray object of rays originating from the same point with Nx by Ny by D direction vector,
    where D is vector dimension (usually 3).
    """
    x_array = sourceplane.x * torch.linspace(-1.0, 1.0, Nx).view(Nx, 1, 1)
    y_array = sourceplane.y * torch.linspace(-1.0, 1.0, Ny).view(1, Ny, 1)
    direction = unit(sourceplane.normal + x_array + y_array)
    return Ray(sourceplane.position_m, direction, **raykwargs)


def ideal_lens(in_ray, lens, f):
    """Ideal infinitely thin lens.

    Input
    -----
        in_ray:  Ray object. Input Ray.
        lens:    Plane object. Lens plane.
        f:       Scalar array. Focal distance.

    Output
    ------
        out_ray: Ray object. Output Ray.

    """

    # Flip lens normal if in_ray is coming from opposite direction (required for backpropagation)
    normal = -lens.normal * torch.sign(dot(lens.normal, in_ray.direction))

    # Define useful points for lens
    L = lens.position_m                                     # Lens position
    F1 = L + f*normal                                       # Front focal point
    F2 = L - f*normal                                       # Back focal point

    # Compute outgoing Ray position and direction
    S_Ray = in_ray.intersect_plane(Plane(F1, normal))       # Propagate Ray to Front focal plane
    P = S_Ray.intersect_plane(lens).position_m              # Intersection with lens plane
    out_dir = unit(L - S_Ray.position_m)                    # Out Ray direction

    # Compute pathlength
    PW_distance_m = dot(out_dir, F2 - P)                    # Distance P to W
    new_pathlength_m = S_Ray.pathlength_m \
        + (2*f - PW_distance_m)                             # Compute pathlength

    ##################
    # # Pathlength paraxial approximation (parabolic wavefront correction)
    # new_pathlength_m = in_ray.intersect_plane(lens).pathlength_m \
    #                  - in_ray.refractive_index * norm_square(P - L) / (2*f)

    # # Pathlength correction spherical wavefront
    # new_pathlength_m = in_ray.intersect_plane(lens).pathlength_m \
    #                  - in_ray.refractive_index * (torch.sqrt(f*f + norm_square(P-L)) - f)
    ##################

    # Return outgoing Ray
    return S_Ray.copy(position_m=P, direction=out_dir, pathlength_m=new_pathlength_m)

#    OC          = lens.position_m                           # Optical Center position
#    BFP         = Plane(OC - f * lens.normal, lens.normal)  # Back Focal Plane
#    chiefray    = Ray(OC, in_ray.direction)                 # Chief or Principal Ray (through OC)
#    focus       = chiefray.intersect_plane(BFP).position_m  # Ray intersection with Back Focal Plane
#
#    intersect   = in_ray.intersect_plane(lens)              # Ray intersection with lens plane
#    newpos      = intersect.position_m                      # Position of the new ray
#    newpath     = intersect.pathlength_m                    # Pathlength of new ray at lens plane
#
#    ### Compute pathlength
#    # Rays further from the axis are delayed less
#
#    chiefraylength = norm(focus - OC) - dot(newpos - OC, chiefray.direction)
#    thisraylength = norm(focus - newpos)
#
#    newpath += chiefraylength - thisraylength
#
#    out_ray = in_ray.copy(position_m=newpos, pathlength_m=newpath,
#        direction=unit(focus - newpos) )    # Output Ray
#    return out_ray


def smooth_grid(xy, power):
    """Smooth Grid function.

    Input
    -----
    """
    x, y = xy.unbind(-1)
    cosinegrid, indices = torch.max(torch.stack((
        (0.5 * torch.cos(x * 2 * np.pi) + 0.5),
        (0.5 * torch.cos(y * 2 * np.pi) + 0.5)), dim=-1), dim=-1)

    return 1 - cosinegrid.unsqueeze(-1) ** power


def intensity_mask_smooth_grid(in_ray, coordplane, power):
    """Intensity Mask of Smooth Grid.

    Pass rays through intensity mask.
    """
    rays_at_plane = in_ray.intersect_plane(coordplane)
    xy = coordplane.transform(rays_at_plane)
    factors = smooth_grid(xy, power)
    return rays_at_plane.mask(factors)


def snells(ray_in, N, n_out):
    """
    Compute refracted Ray.

    Returns a new Ray instance with the new direction and refractive index.

    Input
    -----
        ray_in      Ray. Incoming Ray.
        N           Unit vector. Surface normal of interface.
        n_out       Refractive index outgoing ray

    Output
    ------
        ray_out     Ray. Refracted outgoing Ray.

    Notes on derivation: rejection(ray_dir, surface_normal) has magnitude sin(angle)
    and thus it can act as sine in Snell's law. Furthermore, it is perpendicular to
    the surface_normal. Hence, it's the perpendicular component of dir_out. Once
    this is known, the parallel component can also be computed, since |dir_out| = 1.

    By the same equations, backwards propagating rays can also be computed by
    reversing the surface normal. Make sure the refractive indices of both the Rays
    and the interface are correct though.
    """
    dir_in = ray_in.direction
    n_in = ray_in.refractive_index

    dir_inrej = rejection(dir_in, N)       # Perpendicular component dir_in
    dir_outrej = n_in/n_out * dir_inrej    # Perpendicular component dir_out
    dir_out = dir_outrej - N * torch.sqrt(1 - norm_square(dir_outrej))

    ray_out = ray_in.copy(direction=dir_out, refractive_index=n_out)
    return ray_out


def mirror(ray_in, mirror_plane):
    """
    Mirror.

    Take a ray as input and output the reflected ray.

    Input
    -----
        ray_in          Ray. Input ray.
        mirror_plane    Plane. Mirror plane.

    Output
    ------
        ray_out         Ray. Reflected ray.
    """
    ray_out = ray_in.intersect_plane(mirror_plane)
    new_dir = reflection(ray_in.direction, mirror_plane.normal)
    ray_out.direction = new_dir
    return ray_out


def galvo_mirror(ray_in, galvo_plane, rotations):
    """
    Galvo mirror element that changes direction of ray.

    Take ray as input and reflect off a collection of rotated mirrors. The
    mirror can rotate along the x and y axes of the given galvo Plane.
    Note: rotation along x axis is applied before rotation along y axis.

    Input
    -----
        ray_in      Ray. Input ray. The output ray will copy all properties
                    except direction. The rays will reflect off the tilted
                    galvo mirror.
        galvo_plane CoordPlane. The x and y vectors of the galvo plane represent
                    the rotation axes of the galvo mirrors.
                    galvo response in radians/volt. ##################
        rotations   MxNx2 Tensor. The rotations [x, y] applied to the galvo mirror.

    Output
    ------
        ray_out     Ray. The output ray will copy all properties from ray_in
                    except direction, which will be adjusted according to the
                    reflection.
    """
    assert checkunitvector(galvo_plane.x), 'x of galvo CoordPlane must be unit vector'
    assert checkunitvector(galvo_plane.y), 'y of galvo CoordPlane must be unit vector'

    rot_x, rot_y = components(rotations)        # Split vector dimensions
    mirror_normal_rotx = rotate(galvo_plane.normal, unit(galvo_plane.x), rot_x)
    mirror_normal = rotate(mirror_normal_rotx, unit(galvo_plane.y), rot_y)
    mirror_plane = Plane(galvo_plane.position_m, mirror_normal)
    ray_out = mirror(ray_in, mirror_plane)
    return ray_out


def slm_segment(ray_in, slm_plane, slm_coords):
    """
    SLM segment that sets position of ray.

    Take ray as input and replace its position(s) with positions generated by
    an SLM plane and a set of relative SLM coordinates. Note that the position
    of the input Ray will be completely discarded.

    Input
    -----
        ray_in      Ray. Input ray. The output ray will copy all properties
                    except position_m.
        SLM_plane   CoordPlane. The x and y vectors of the SLM plane represent
                    the physical size of the SLM from edge to edge.
        SLM_coords  Tensor. Coordinates of the output ray positions in relative
                    SLM coordinates, i.e. ranging from top to bottom = [-0.5, 0.5].
                    Aspect ratio = 1.

    Output
    ------
        ray_out     Ray. The output ray will copy all properties from ray_in
                    except position_m, which will be generated from the
                    SLM data.
    """
    x_slm, y_slm = components(slm_coords)    # Split vector dimensions
    position_m = slm_plane.position_m + slm_plane.x * x_slm + slm_plane.y * y_slm
    ray_out = ray_in.copy(position_m=position_m)
    return ray_out


#### Create camera function that includes ray-plane intersection and coordinate transformation
