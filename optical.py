"""Optical functions.

Functions that represent optical elements. They have Ray objects as input and/or output.

For clarification of the meaning in this context of terminology like vectors and scalars,
see vector_functions.py.
"""

import torch
import numpy as np

from testing import checkunitvector, machine_epsilon_f64
from vector_functions import unit, dot, norm_square, rejection, reflection, rotate, components, \
    cartesian3d, Scalar
from math_functions import solve_quadratic, sign
from ray_plane import Ray, CoordPlane, Plane, translate, copy_update, PlaneType
from plot_functions import default_viewplane, plot_plane


class OpticalSystem():
    """
    Template class for an optical system.
    """
    def __init__(self):
        pass

    def update(self):
        pass

    def raytrace(self, in_ray):
        rays = []
        return rays

    def backtrace(self, in_ray):
        rays = []
        return rays

    def plot(self, ax, viewplane):
        pass


class Coverslip(OpticalSystem):
    """
    Coverslip as optical system.

    Coverslip front plane normal is pointed in direction of back plane.
    """
    def __init__(self):
        super().__init__()
        origin, x, y, z = cartesian3d()
        self.coverslip_thickness_m = 170e-6
        self.n_coverslip = 1.5185
        self.n_out_backside = 1.0
        self.n_out_frontside = 1.3304
        self.sample_plane = CoordPlane(origin, x, y)

    def update(self):
        self.coverslip_front_plane = copy_update(self.sample_plane)
        self.coverslip_back_plane = translate(self.coverslip_front_plane, self.coverslip_thickness_m
                                              * self.coverslip_front_plane.normal)
        self.desired_focus_plane = copy_update(self.coverslip_back_plane)

    def raytrace(self, in_ray):
        rays = [flat_interface(in_ray, self.coverslip_front_plane, self.n_coverslip)]
        rays += [flat_interface(rays[-1], self.coverslip_back_plane, self.n_out_backside)]
        return rays

    def backtrace(self, in_ray):
        rays = [flat_interface(in_ray, self.coverslip_back_plane, self.n_coverslip)]
        rays += [flat_interface(rays[-1], self.coverslip_front_plane, self.n_out_frontside)]
        return rays

    def plot(self, ax, viewplane=default_viewplane(), plotkwargs={'color': 'black'}):
        plot_scale = 5e-3
        plot_plane(ax, self.coverslip_front_plane, text1='Coverslip\nfront', scale=plot_scale,
                   viewplane=viewplane, plotkwargs=plotkwargs)
        plot_plane(ax, self.coverslip_back_plane, text2='Coverslip\nback', scale=plot_scale,
                   viewplane=viewplane, plotkwargs=plotkwargs)
        pass


def collimated_source(source_plane, Nx, Ny, **raykwargs):
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

    Note: if Nx==1 or Ny==1, the corresponding ray elements will be centered at the sourceplane
    position for that dimension.

    Output
    ------
    A Ray object of collimated rays with Nx by Ny by 3 position_m. Direction is defined by
    sourceplane normal vector.
    """
    x_array = source_plane.x * torch.linspace(-1.0, 1.0, Nx).view(Nx, 1, 1) * (Nx != 1)
    y_array = source_plane.y * torch.linspace(-1.0, 1.0, Ny).view(1, Ny, 1) * (Ny != 1)
    position = source_plane.position_m + x_array + y_array
    return Ray(position, source_plane.normal, **raykwargs)


def point_source(source_plane, Nx, Ny, **raykwargs):
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

    Note: if Nx==1 or Ny==1, the corresponding ray elements will be centered at the sourceplane
    position for that dimension.

    Output
    ------
    A Ray object of rays originating from the same point with Nx by Ny by D direction vector,
    where D is vector dimension (usually 3).
    """
    x_array = source_plane.x * torch.linspace(-1.0, 1.0, Nx).view(Nx, 1, 1) * (Nx != 1)
    y_array = source_plane.y * torch.linspace(-1.0, 1.0, Ny).view(1, Ny, 1) * (Ny != 1)
    direction = unit(source_plane.normal + x_array + y_array)
    return Ray(source_plane.position_m, direction, **raykwargs)


def thin_lens(in_ray: Ray, lens: PlaneType, f: Scalar) -> Ray:
    """
    Thin lens
    Follows height = tan(angle). Pathlength is corrected for point sources at the front focal plane.
    In other cases, only works for paraxial rays. Rays are 'refracted' such that the thin lens law
    1/f = 1/s1 + 1/s2 works (where f=focal distance, s1 and s2 = object and image distance to lens).

    todo: Explain the steps of computation in a drawing in some documentation.

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
    propagation_sign = sign(dot(lens.normal, lens.position_m - in_ray.position_m))
    normal = -lens.normal * propagation_sign

    # Define useful points for lens
    L = lens.position_m                                     # Lens position
    F1 = L + f*normal                                       # Front focal point
    F2 = L - f*normal                                       # Back focal point

    # Compute outgoing Ray position and direction
    S_Ray = in_ray.intersect_plane(Plane(F1, normal))       # Propagate Ray to Front focal plane
    P = S_Ray.intersect_plane(lens).position_m              # Intersection with lens plane
    out_dir = unit(L - S_Ray.position_m)                    # Out Ray direction

    # Compute pathlength correction for point sources at front focal plane
    PW_distance_m = dot(out_dir, F2 - P)                    # Distance P to W
    new_pathlength_m = S_Ray.pathlength_m \
        + propagation_sign * (2*f - PW_distance_m)          # Compute pathlength

    # Return outgoing Ray
    return copy_update(S_Ray, position_m=P, direction=out_dir, pathlength_m=new_pathlength_m)


def abbe_lens(in_ray, lens, f, n_out=1.0):
    """
    Abbe lens
    Conforms to the Abbe sine condition *for each point at both focal planes*:
    height_out ∝ sin(angle_in) and sin(angle_out) ∝ height_in. Pathlength is fully corrected.

    todo: Explain the steps of computation in a drawing in some documentation.

    Input
    -----
        in_ray:  Ray object. Input Ray.
        lens:    Plane object. Lens plane.
        f:       Scalar array. Focal distance.
        n_out:   Refractive index of medium on other side of lens.

    Note: all length quantities must be in the same unit.

    Output
    ------
        Tuple of Ray objects representing Rays on principal spheres. The first Ray is an
        intermediate Ray. The second Ray is the outgoing Ray.
    """

    # Determine propagation direction (forward or backward)
    propagation_sign = sign(dot(lens.normal, lens.position_m - in_ray.position_m)
                                  * dot(lens.normal, in_ray.direction))

    # Flip lens normal if in_ray is coming from opposite direction
    normal = -lens.normal * sign(dot(lens.normal, lens.position_m - in_ray.position_m))

    # Define useful points for lens
    L = lens.position_m                                     # Lens position
    F1 = L + f*normal                                       # Front focal point
    F2 = L - f*normal                                       # Back focal point

    # Compute points on principal spheres
    S_ray = in_ray.intersect_plane(Plane(F1, normal))       # Propagate Ray to Front focal plane
    in_dir = in_ray.direction * propagation_sign            # Incoming direction towards lens
    n_in = in_ray.refractive_index                          # Refractive index for incoming ray
    SF1 = S_ray.position_m - F1                             # Vector from S to F1

    # Position and pathlength at Back focal plane
    Q = F2 + rejection(in_dir, normal) * f * n_in
    pathlength_SQ = f * (n_in + n_out) - dot(in_dir, SF1) * n_in
    new_pathlength = S_ray.pathlength_m + pathlength_SQ * propagation_sign

    # Outgoing direction (forward) at Back focal plane
    out_dir_rej = -SF1 / (n_out * f)                        # Perpendicular component
    out_dir_proj = - normal * torch.sqrt(1 - norm_square(out_dir_rej))  # Parallel component

    if out_dir_proj.isnan().any():
        pass

    out_dir = out_dir_rej + out_dir_proj                    # Outgoing direction (forward)

    # Outgoing Ray at Back focal plane
    Q_ray = copy_update(in_ray, position_m=Q, direction=out_dir*propagation_sign,
                        pathlength_m=new_pathlength, refractive_index=n_out)

    # Rays at principal spheres
    P1_ray = S_ray.propagate(f*propagation_sign)
    P2_ray = Q_ray.propagate(-f*propagation_sign)

    if P1_ray.position_m.isnan().any():
        pass

    if P2_ray.position_m.isnan().any():
        pass

    return (P1_ray, P2_ray)


def coverslip_correction(in_ray, normal, coverslip_thickness_m, n_coverslip, n_out,
                         propagation_sign):
    """
    Coverslip correction.

    Can be used e.g. in combination with an Abbe sine lens function, to incorporate the coverslip
    correction of an objective. The coverslip correction is implemented by backtracing the effect of
    a coverslip. This function effectively shifts the Ray positions along a plane, depending on the
    Ray directions and the given coverslip properties.

    Input
    -----
        in_ray:                 Ray object. Input Ray.
        normal:                 Normal unit vector of coverslip Plane.
        coverslip_thickness_m:  Thickness of coverslip.
        n_coverslip:            Refractive index of coverslip.
        n_out:                  Refractive index of medium of outgoing Ray.
        propagation_sign:       Either 1 for forward or -1 for backward propagation.

    Output
    ------
        out_ray:                Ray object. Ray after coverslip correction.
    """

    assert checkunitvector(normal)

    # Get a Plane normal in opposite direction of propagation
    N = -normal * propagation_sign * sign(dot(normal, in_ray.direction))

    # Create a coverslip plane located 1 coverslip thickness behind the ray
    coverslip_front_plane = Plane(in_ray.position_m, N)
    coverslip_back_plane = translate(coverslip_front_plane, N * coverslip_thickness_m)

    # Ray trace
    ray1 = snells(in_ray, N, n_coverslip)                       # Refract to coverslip medium
    ray2 = flat_interface(ray1, coverslip_back_plane, n_out)    # Coverslip 'back' interface
    out_ray = ray2.intersect_plane(coverslip_front_plane)       # Propagate back to original plane

    return out_ray


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


def snells(ray_in, normal, n_out):
    """
    Compute refracted Ray.

    Returns a new Ray instance with the new direction and refractive index.

    Input
    -----
        ray_in      Ray. Incoming Ray.
        normal      Unit vector. Surface normal of interface.
        n_out       Refractive index outgoing ray

    Output
    ------
        ray_out     Ray. Refracted outgoing Ray.

    Notes on derivation: rejection(ray_dir, surface_normal) has magnitude sin(angle)
    and thus it can act as sine in Snell's law. Furthermore, it is perpendicular to
    the surface_normal. Hence, it's the perpendicular component of dir_out. Once
    this is known, the parallel component can also be computed, since |dir_out| = 1.

    This function works regardless of incoming ray direction. If the ray is coming
    from the opposite direction, the normal vector will be flipped to make the computation work.
    If the angle is so large such that total internal reflection would occur, the argument to
    to torch.sqrt will be < 0, resulting in a nan for the output.

    See also:
    https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
    """

    assert checkunitvector(normal)

    # Flip normal if ray_in is coming from opposite direction (required for backpropagation)
    N = -normal * sign(dot(normal, ray_in.direction))

    dir_in = ray_in.direction
    n_in = ray_in.refractive_index

    dir_inrej = rejection(dir_in, N)        # Perpendicular component (i.e. along plane) of dir_in
    dir_outrej = n_in/n_out * dir_inrej     # Perpendicular component (i.e. along plane) of dir_out
    dir_outproj = - N * torch.sqrt(1 - norm_square(dir_outrej))     # Parallel component of dir_out

    if dir_outproj.isnan().any():
        pass

    dir_out = dir_outrej + dir_outproj      # Combine components

    # mask_TIR = (norm_square(dir_outrej) > 1)        # Scalar mask Total Internal Reflection occurs
    # mask_TIR_vec = mask_TIR.expand_as(dir_out)      # Vector mask Total Internal Reflection occurs
    # dir_out[mask_TIR_vec] = dir_in.expand_as(dir_out)[mask_TIR_vec]    # These dir_outs are useless due to TIR

    ray_out = copy_update(ray_in, direction=dir_out, refractive_index=n_out)
    # ray_out.weight = ray_out.weight * mask_TIR.logical_not()    # Set weight to zero when TIR occurs

    return ray_out


def flat_interface(in_ray, interface_plane, n_new):
    """
    Flat Interface
    Intersect with flat plane interface and refract.

    Input
    -----
    in_ray              Ray. Incoming ray.
    plane               Plane. Defines the plane of the flat interface.
    n_new               Scalar. Refractive index of refracted, outgoing Ray.

    Output
    ------
    out_ray             Ray. Refracted, outgoing Ray.
    """
    ray_at_plane = in_ray.intersect_plane(interface_plane)
    out_ray = snells(ray_at_plane, interface_plane.normal, n_new)
    return out_ray


def propagate_to_cylinder(in_ray, cylinder_plane, radius_m, propagation_sign=1):
    """
    Cylinder Interface
    Intersect with cylinder with arbitrary orientation.

    Input
    -----
    in_ray              Ray. Incoming ray.
    cylinder_plane      CoordPlane. Defines the cylinder direction with its normal vector. x and y
                        must be unit vectors.
    radius_m            Scalar. Radius of the cylinder.
    propagation_sign    Scalar: either 1 or -1. Defines the propagation direction (1 = forward,
                        -1 = backward).

    Output
    ------
    out_ray             Ray. Outgoing ray, with position at cylinder interface.

    Notes
    -----
    For derivation see #####
    """

    checkunitvector(cylinder_plane.x)
    checkunitvector(cylinder_plane.y)

    # Position and direction vectors, projected on cylinder plane, in coords of cylinder plane
    Pxy = cylinder_plane.transform_points(in_ray.position_m)
    Dxy = cylinder_plane.transform_direction(in_ray.direction)

    # Quadratic equation coefficients
    a = norm_square(Dxy)
    b = 2 * dot(Dxy, Pxy)
    c = norm_square(Pxy) - radius_m*radius_m

    # Compute distances to intersections (with sign in propagation direction)
    distances_m = torch.cat(solve_quadratic(a, b, c), -1) * propagation_sign

    if distances_m.isnan().any():
        pass

    # Nearest intersection distance (in propagation direction)
    distances_m[distances_m < 100*machine_epsilon_f64] = torch.inf      # Interface at or behind ray
    nearest_distance_m = distances_m.min(dim=-1, keepdim=True).values   # Nearest interface
    nearest_distance_m[nearest_distance_m == torch.inf] = 0.0   # No interface in front of ray
    nearest_distance_m[nearest_distance_m.isnan()] = 0.0        # Rays that miss cylinder entirely

    # Propagate ray to new position
    out_ray = in_ray.propagate(nearest_distance_m * propagation_sign)
    return out_ray


def cylinder_interface(in_ray, cylinder_plane, radius_m, n_new, propagation_sign=1): ####### prop sign should have no default
    """
    Cylinder Interface
    Intersect and refract ray with cylinder with arbitrary orientation.

    Input
    -----
    in_ray              Ray. Incoming ray.
    cylinder_plane      CoordPlane. Defines the cylinder direction with its normal vector.
    radius_m            Scalar. Radius of the cylinder.
    n_new               Scalar. Refractive index of medium after refraction.
    propagation_sign    Scalar: either 1 or -1. Defines the propagation direction (1 = forward,
                        -1 = backward).

    Output
    ------
    out_ray         Ray. Propagated and refracted outgoing ray.
    """
    # Propagate to cylinder interface
    ray_at_cylinder = propagate_to_cylinder(in_ray, cylinder_plane, radius_m, propagation_sign)

    # Compute cylinder normal
    Q = ray_at_cylinder.position_m
    C = cylinder_plane.position_m
    N = cylinder_plane.normal
    normal = unit(rejection(Q-C, N))

    # Refract Ray
    out_ray = snells(ray_at_cylinder, normal, n_new)
    return out_ray


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
        rotations   MxNx2 Tensor. The mechanical rotations [x, y] applied to the
                    galvo mirror.

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
    ray_out = copy_update(ray_in, position_m=position_m)
    return ray_out
