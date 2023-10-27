"""
Lumen-based organ-on-a-chip Sample.
"""
from torch import Tensor

from ray_plane import CoordPlane, translate, copy_update
from vector_functions import cartesian3d, rotate
from optical import OpticalSystem, cylinder_interface, flat_interface
from materials import n_water, n_SodaLimeGlass, n_hydrated_collagen, n_PBS
from plot_functions import plot_cylinder, plot_plane, default_viewplane


class SampleLumenChip(OpticalSystem):
    """
    A lumen-based organ-on-a-chip sample with a tube in collagen, sitting on a glass microscope
    slide. The collagen is held in place by PDMS. The chips will be imaged from the bottom, meaning
    the PDMS does not need to be modeled. The cylindrical tube is filled with PBS.

    A wavelength must be set to define the refractive index properties of the materials. The
    .update() method updates planes A, B and C based on the attributes (radius, thickness, etc.).

    We define 5 planes to describe this sample:
     - Plane A: flat water-glass interface
     - Plane B: flat glass-collagen interface
     - Plane C: plane defining the cylinder of collagen-PBS interface.
     - Sample plane: defines the position and orientation of the sample. It may be externally
        modified. The rest of the sample translates and rotates with this plane.
     """
    def __init__(self):
        super().__init__()
        origin, x, y, z = cartesian3d()

        # Attributes
        self.radius_m = Tensor((200e-6,))               # Radius of tube
        self.slide_thickness_m = Tensor((1e-3,))        # Thickness of glass microscope slide
        self.collagen_thickness_m = Tensor((100e-6,))   # Tube edge to glass slide
        self.tube_yaw = Tensor((0.,))                   # Angle of tube around optical axis
        self.tube_pitch = Tensor((0.,))                 # Angle of tube around x

        self.sample_plane = CoordPlane(origin, x, y)    # Initial sample position

        # Properties
        self.__wavelength_m = 1e-6                      # Wavelength in meter
        self.__collagen_fiber_fraction = 0.36           # Taken from [Leonard and Meek 1997] 'f_m'

    @property
    def wavelength_m(self):
        """Get the wavelength that was set for the material properties."""
        return self.__wavelength_m

    @wavelength_m.setter
    def wavelength_m(self, wavelength_m):
        """Set the wavelength to specify material properties."""
        self.__wavelength_m = wavelength_m
        self._n_water = n_water(wavelength_m)
        self._n_slide = n_SodaLimeGlass(wavelength_m)
        self._n_collagen = n_hydrated_collagen(wavelength_m, self.collagen_fiber_fraction)
        self._n_PBS = n_PBS(wavelength_m)

    @property
    def collagen_fiber_fraction(self):
        """Get the collagen fiber volume fraction, used to define collagen properties."""
        return self.__collagen_fiber_fraction

    @collagen_fiber_fraction.setter
    def collagen_fiber_fraction(self, new_fraction):
        """Set the collagen fiber volume fraction, used to define collagen properties."""
        self.__collagen_fiber_fraction = new_fraction
        self._n_collagen = n_hydrated_collagen(self.__wavelength_m, self.__collagen_fiber_fraction)

    def update(self):
        """
        Define/update planes A, B, C.
        """
        # Plane C - cylinder plane
        x_rot = rotate(self.sample_plane.x, self.sample_plane.normal, self.tube_yaw)
        y_rot = rotate(self.sample_plane.normal, x_rot, self.tube_pitch)
        self._plane_C = CoordPlane(self.sample_plane.position_m, x_rot, y_rot)

        # Plane B - glass-collagen interface
        translation_B = (self.radius_m + self.collagen_thickness_m) * -self.sample_plane.normal
        self._plane_B = translate(self.sample_plane, translation_B)

        # Plane A - water-glass interface
        translation_A = self.slide_thickness_m * -self._plane_B.normal
        self._plane_A = translate(self._plane_B, translation_A)

    def raytrace(self, in_ray):
        rays = []
        raise NotImplementedError('Ray trace method currently undefined.')
        return rays

    def backtrace(self, in_ray):
        """Trace rays from within the tube, through the collagen and glass."""
        rays = []
        rays += [copy_update(in_ray, refractive_index=self._n_PBS)]             # Start in PBS
        rays += [cylinder_interface(rays[-1], self._plane_C, self.radius_m, self._n_collagen)]
        rays += [flat_interface(rays[-1], self._plane_B, n_new=self._n_slide)]  # Plane B
        rays += [flat_interface(rays[-1], self._plane_A, n_new=self._n_water)]  # Plane A
        return rays

    def plot(self, ax, viewplane=default_viewplane(), plotkwargs={'color': 'black'}):
        """Plot sample."""
        plot_scale = 8
        plot_cylinder(ax, self._plane_C, self.radius_m, plot_scale*self.radius_m, 0,
                      viewplane=viewplane, plotkwargs=plotkwargs)
        plot_plane(ax, self._plane_B, scale=plot_scale*200e-6,
                   viewplane=viewplane, plotkwargs=plotkwargs)
        plot_plane(ax, self._plane_A, scale=plot_scale*200e-6,
                   viewplane=viewplane, plotkwargs=plotkwargs)
