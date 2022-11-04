"""
Tube Sample. Cylindrical tube sample class.
"""
from torch import Tensor

from ray_plane import CoordPlane
from vector_functions import cartesian3d, rotate
from optical import OpticalSystem, cylinder_interface, flat_interface
from plot_functions import plot_cylinder, plot_plane, default_viewplane


class SampleTube(OpticalSystem):
    """
    Tube Sample. Cylindrical tube sitting on a microscope slide, sample class. Tube direction is
    defined in the top slide plane y direction (for angle=0).
    """
    def __init__(self):
        super().__init__()
        origin, x, y, z = cartesian3d()
        self.inner_radius_m = Tensor((350e-6,))
        self.outer_radius_m = Tensor((500e-6,))
        self.slide_thickness_m = Tensor((1e-3,))
        self.tube_angle = Tensor((0.,))     # Angle of tube around slide normal vector
        self.n_tube = 1.5127                # Schott N-BK7 @715nm
        self.n_slide = 1.5191               # Soda Lime Glass @715nm
        self.n_inside = 1.3304              # Water inside the tube @715nm
        self.n_outside = 1.3304             # Water between tube and slide @715nm
        self.slide_top_plane = CoordPlane(origin, x, y)

    def update(self):
        self.slide_bottom_plane = CoordPlane(
            self.slide_top_plane.position_m - 2 * self.outer_radius_m * self.slide_top_plane.normal,
            self.slide_top_plane.x, self.slide_top_plane.y)
        self.cyl_plane = CoordPlane(
            self.slide_top_plane.position_m + self.outer_radius_m * self.slide_top_plane.normal,
            rotate(self.slide_top_plane.x, self.slide_top_plane.normal, self.tube_angle),
            self.slide_top_plane.normal)    # Slide y == Cyl. plane normal

    def raytrace(self, in_ray):
        rays = []
        rays += [cylinder_interface(in_ray, self.cyl_plane, self.outer_radius_m, self.n_tube)]
        rays += [cylinder_interface(rays[-1], self.cyl_plane, self.inner_radius_m, self.n_inside)]
        rays += [cylinder_interface(rays[-1], self.cyl_plane, self.inner_radius_m, self.n_tube)]
        rays += [cylinder_interface(rays[-1], self.cyl_plane, self.outer_radius_m, self.n_outside)]
        rays += [flat_interface(rays[-1], self.slide_top_plane, self.n_slide)]
        rays += [rays[-1].intersect_plane(self.slide_bottom_plane)]
        return rays

    def backtrace(self, in_ray):
        rays = []
        return rays

    def plot(self, ax, viewplane=default_viewplane(), plotkwargs={'color': 'black'}):
        plot_scale = 4
        plot_cylinder(ax, self.cyl_plane, self.inner_radius_m, 2*plot_scale*self.outer_radius_m, 0,
                      viewplane=viewplane, plotkwargs=plotkwargs)
        plot_cylinder(ax, self.cyl_plane, self.outer_radius_m, 2*plot_scale*self.outer_radius_m, 0,
                      viewplane=viewplane, plotkwargs=plotkwargs)
        plot_plane(ax, self.slide_top_plane, scale=plot_scale*self.outer_radius_m,
                   viewplane=viewplane, plotkwargs=plotkwargs)
        plot_plane(ax, self.slide_bottom_plane, scale=plot_scale*self.outer_radius_m,
                   viewplane=viewplane, plotkwargs=plotkwargs)
