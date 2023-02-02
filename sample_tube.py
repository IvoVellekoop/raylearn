"""
Tube Sample. Cylindrical tube sample class.
"""
from torch import Tensor

from ray_plane import CoordPlane, copy_update, translate
from vector_functions import cartesian3d, rotate
from optical import OpticalSystem, cylinder_interface, flat_interface, point_source
from plot_functions import plot_cylinder, plot_plane, default_viewplane


class SampleTube(OpticalSystem):
    """
    Tube Sample. Cylindrical tube sitting on a microscope slide, sample class. Tube direction is
    defined in the top slide plane y direction (for angle=0). The slide top plane is defined at the
    sample plane, but with the normal pointing in the opposite direction.
    """
    def __init__(self):
        super().__init__()
        origin, x, y, z = cartesian3d()
        self.shell_thickness_m = Tensor((135e-6,))
        self.outer_radius_m = Tensor((485e-6,))
        self.slide_thickness_m = Tensor((1e-3,))
        self.tube_angle = Tensor((0.,))     # Angle of tube around slide normal vector
        self.n_tube = 1.5127                # Schott N-BK7 @715nm
        self.n_slide = 1.5191               # Soda Lime Glass @715nm
        self.n_inside = 1.3304              # Water inside the tube @715nm
        self.n_outside = 1.3304             # Water between tube and slide @715nm
        self.sample_plane = CoordPlane(origin, x, y)

    def update(self):
        self.inner_radius_m = self.outer_radius_m - self.shell_thickness_m
        self.slide_top_plane = copy_update(self.sample_plane, x=-self.sample_plane.x)
        self.slide_bottom_plane = translate(self.slide_top_plane,
                                            -self.slide_thickness_m*self.slide_top_plane.normal)
        self.cyl_plane = CoordPlane(
            self.slide_top_plane.position_m + self.outer_radius_m * self.slide_top_plane.normal,
            rotate(self.slide_top_plane.x, self.slide_top_plane.normal, self.tube_angle),
            self.slide_top_plane.normal)    # Slide y == Cyl. plane normal

        shell_thickness = self.outer_radius_m - self.inner_radius_m
        self.desired_focus_plane = translate(
            self.slide_top_plane, shell_thickness * self.slide_top_plane.normal)

    def raytrace(self, in_ray):
        rays = []
        rays += [cylinder_interface(in_ray, self.cyl_plane, self.outer_radius_m, self.n_tube)]
        rays += [cylinder_interface(rays[-1], self.cyl_plane, self.inner_radius_m, self.n_inside)]
        rays += [cylinder_interface(rays[-1], self.cyl_plane, self.inner_radius_m, self.n_tube)]
        self.rays_at_desired_focus = rays[-1]
        rays += [cylinder_interface(rays[-1], self.cyl_plane, self.outer_radius_m, self.n_outside)]
        rays += [flat_interface(rays[-1], self.slide_top_plane, self.n_slide)]
        rays += [rays[-1].intersect_plane(self.slide_bottom_plane)]
        return rays

    def backtrace(self, in_ray):
        rays = []
        rays += [cylinder_interface(in_ray, self.cyl_plane, self.inner_radius_m, self.n_tube, propagation_sign=1)]
        rays += [cylinder_interface(rays[-1], self.cyl_plane, self.outer_radius_m, self.n_outside, propagation_sign=1)]
        return rays

    def plot(self, ax, viewplane=default_viewplane(), plotkwargs={'color': 'black'}):
        plot_scale = 2
        plot_cylinder(ax, self.cyl_plane, self.inner_radius_m, 4*plot_scale*self.outer_radius_m, 0,
                      viewplane=viewplane, plotkwargs=plotkwargs)
        plot_cylinder(ax, self.cyl_plane, self.outer_radius_m, 4*plot_scale*self.outer_radius_m, 0,
                      viewplane=viewplane, plotkwargs=plotkwargs)
        plot_plane(ax, self.slide_top_plane, scale=plot_scale*self.outer_radius_m,
                   viewplane=viewplane, plotkwargs=plotkwargs)
        plot_plane(ax, self.slide_bottom_plane, scale=plot_scale*self.outer_radius_m,
                   viewplane=viewplane, plotkwargs=plotkwargs)
