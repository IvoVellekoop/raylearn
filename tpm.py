"""
The Two Photon Microscope.
"""

import torch
from torch import tensor, Tensor
import numpy as np
import matplotlib.pyplot as plt

from vector_functions import rotate, cartesian3d
from ray_plane import Ray, Plane, CoordPlane, copy_update
from plot_functions import plot_plane, plot_lens, plot_rays, default_viewplane
from optical import OpticalSystem, point_source, thin_lens, abbe_lens, snells, galvo_mirror, \
    slm_segment, flat_interface, coverslip_correction


# Set default tensor type to double (64 bit)
# Machine epsilon of float (32-bit) is 2^-23 = 1.19e-7
# The ray simulation contains both meter and micrometer scales,
# hence floats might not be precise enough.
# https://en.wikipedia.org/wiki/Machine_epsilon
torch.set_default_tensor_type('torch.DoubleTensor')

plt.rc('font', size=12)


class TPM(OpticalSystem):
    """
    The Two Photon Microscope.

    Conveniently pack all variables that make up the TPM in one class.
    """

    def __init__(self):
        """
        Define all properties of all optical elements as initial guess.

        All statically defined properties should be defined here.
        Properties that depend on other properties should be computed in
        the update method. All lengths in meters.

        The measurements define the SLM segment and Galvo Mirror settings,
        and are hence included.

        Note: For now, the TPM alignment up to the SLM is assumed to be
        perfect. The magnification between Galvo Mirrors and the SLM is 1.
        Hence the Galvo Mirrors and SLM are all defined on the same position,
        skipping the 4F system lenses in between them.
        """
        super().__init__()

        # Define refractive indices at wavelength of 715nm
        # References:
        # https://refractiveindex.info/?shelf=main&book=H2O&page=Hale
        # https://refractiveindex.info/?shelf=glass&book=soda-lime&page=Rubin-clear
        # https://refractiveindex.info/?shelf=glass&book=SCHOTT-multipurpose&page=D263TECO
        self.n_water = 1.3304
        self.n_obj2_coverslip = 1.5185

        # Define coordinate system
        self.coordsystem = cartesian3d()
        origin, x, y, z = self.coordsystem

        # Galvo Mirrors
        # Thorlabs GVS111(/M)
        # https://bmpi.wiki.utwente.nl/doku.php?id=instrumentation:galvo:galvo_scanners
        # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=7616&pn=GVS111/M#7617
        # Optical scan angle = 2x mechanical angle
        # https://docs.scanimage.org/Configuration/Scanners/Resonant%2BScanner.html
        # https://bmpi.wiki.utwente.nl/lib/exe/fetch.php?media=instrumentation:galvo:gvs111_m-manual.pdf
        # self.galvo_volts_per_optical_degree = 0.5 / 1.81    # Factor of 1.81 measured on 11-05-2022
        self.galvo_volts_per_optical_degree = tensor(0.5) / 2   #############  # Factor of 1.81 measured on 11-05-2022. Now it's 2..?
        # Galvo mirror rotation in mechanical radians per volt
        # conversion from optical scan angle -> factor of 2
        self.galvo_mech_rad_per_V = (np.pi/180) / 2 / self.galvo_volts_per_optical_degree
        self.galvo_roll = tensor((0.0,))                   # Rotation angle around optical axis

        # SLM
        # Meadowlark 1920x1152 XY Phase Series
        # https://bmpi.wiki.utwente.nl/doku.php?id=instrumentation:slm:meadowlark_slm
        # https://www.meadowlark.com/store/data_sheet/SLM%20-%201920%20x%201152%20Data%20Sheet%20021021.pdf
        # https://www.meadowlark.com/images/files/Specification%20Backgrounder%20for%20XY%20Series%20Phase%20SLMS%20-%20SB0520.pdf
        self.slm_height_m = 10.7e-3
        self.slm_height_pixels = 1152
        self.slm_angle  = tensor((0.,))         # Rotation angle around optical axis
        self.slm_zshift = tensor((0.,))

        # Coverslip
        self.total_coverslip_thickness = tensor((170e-6,))
        self.coverslip_tilt_around_x = tensor((0.0,))
        self.coverslip_tilt_around_y = tensor((0.0,))

        # Focal distances (m)
        self.f1 = 100e-3
        self.f2 = 100e-3
        self.f3 = 200e-3
        self.f4 = 200e-3
        self.f5 = 150e-3
        self.f6a = 200e-3
        self.f6b = 200e-3
        self.f7 = 300e-3
        self.f9 = 150e-3
        self.f10 = 40e-3
        # Objective 1: Nikon CFI75 LWD 16X W
        self.obj1_tubelength = 200e-3           # Objective standard tubelength
        self.obj1_magnification = 16            # Objective magnification
        self.fobj1 = self.obj1_tubelength / self.obj1_magnification
        # Objective 2: Zeiss A-Plan 100x/0.8 421090-9800
        self.obj2_tubelength = 165e-3           # Objective standard tubelength
        self.obj2_magnification = 20            # Objective magnification
        self.fobj2 = self.obj2_tubelength / self.obj2_magnification
        self.obj1_NA = 0.8
        self.obj2_NA = 0.8
        self.obj2_coverslip_thickness_m = 170e-6

        # Initial z-shifts of lens planes
        self.obj1_zshift = tensor((0.,))
        self.sample_zshift = tensor((0.,))
        self.obj2_xshift = tensor((0.,))
        self.obj2_yshift = tensor((0.,))
        self.obj2_zshift = tensor((0.,))
        self.L9_zshift = tensor((0.,))
        self.L10_zshift = tensor((0.,))

        # Camera planes
        # Basler acA2000-165umNIR
        # https://www.baslerweb.com/en/products/cameras/area-scan-cameras/ace/aca2000-165umnir/
        self.cam_pixel_size = 5.5147e-6
        self.cam_ft_xshift = tensor((0.,))
        self.cam_ft_yshift = tensor((0.,))
        self.cam_im_xshift = tensor((0.,))
        self.cam_im_yshift = tensor((0.,))
        self.cam_im_zshift = tensor((0.,))

        # Desired focus position relative to the sample plane
        # Note: when moving the desired focus position past an optical element, the raytrace method
        # and backtrace method must be manually adapted accordingly, so the ray plane intersection
        # happens at the correct ray trace step.
        self.backtrace_source_opening_tan_angle = tensor(1.,)   # Is manually adjusted later
        #### Idea: automatically determine when also compensating for focus?

        self.desired_focus_position_relative_to_sample_plane = tensor((0., 0., 0.))
        self.backtrace_Nx = 300
        self.backtrace_Ny = 300

        self.sample = OpticalSystem()

    def set_measurement(self, matfile):
        # SLM coords and Galvo rotations
        self.slm_coords = tensor(matfile['p/rects'])[0:2, :].T.view(-1, 2)
        self.galvo_volts = tensor((matfile['p/galvoXs'], matfile['p/galvoYs'])).T \
                          - tensor((matfile['p/GalvoXcenter'], matfile['p/GalvoYcenter'])).view(1, 1, 2)
        ######### Correct with SLM ppp instead
        self.galvo_rots = self.galvo_volts * self.galvo_mech_rad_per_V

    def update(self):
        """
        Update dependent properties. These properties are defined as a function
        of statically defined properties and/or other dependent properties.

        Properties that depend on dynamic properties should be computed here,
        so they get recomputed whenever update is called. All lengths in meters.
        """
        origin, x, y, z = self.coordsystem

        # Galvo
        self.galvo_x = rotate(-x, z, self.galvo_roll)
        self.galvo_y = rotate(-y, z, self.galvo_roll)
        self.galvo_plane = CoordPlane(origin, self.galvo_x, self.galvo_y)

        # Lens 3 and 4
        self.L3 = Plane(self.galvo_plane.position_m + self.f3 * z, -z)
        self.plane34 = CoordPlane(self.L3.position_m + self.f3 * z, x, y)
        self.L4 = Plane(self.L3.position_m + (self.f3 + self.f4) * z, -z)

        # SLM
        self.slm_x = rotate(x * self.slm_height_m, z, self.slm_angle)
        self.slm_y = rotate(y * self.slm_height_m, z, self.slm_angle)
        self.slm_plane = CoordPlane(self.L4.position_m + (self.f4 + self.slm_zshift) * z,
                                    self.slm_x, self.slm_y)

        # Lens planes to sample plane
        self.L5 = Plane(self.slm_plane.position_m + self.f5*z, -z)
        self.plane57 = CoordPlane(self.L5.position_m + self.f5 * z, x, y)
        self.L7 = Plane(self.L5.position_m + (self.f5 + self.f7)*z, -z)
        self.OBJ1 = Plane(self.L7.position_m + (self.f7 + self.fobj1 + self.obj1_zshift)*z, -z)

        # Sample plane
        self.sample_plane = CoordPlane(self.L7.position_m + (self.f7 + 2*self.fobj1 +
                                       self.sample_zshift) * z, x, y)

        # Objective
        self.OBJ2 = Plane(self.sample_plane.position_m + self.fobj2*z + self.obj2_zshift*z + self.obj2_xshift*x + self.obj2_yshift*y, -z)
        self.L9 = Plane(self.OBJ2.position_m + (self.fobj2 + self.f9 + self.L9_zshift)*z, -z)
        self.L10 = Plane(self.L9.position_m + (self.f9 + self.f10)*z + self.L10_zshift*z, -z)

        # Cameras
        self.cam_im_plane = CoordPlane(self.L9.position_m + self.f9*z +
                                       self.cam_im_xshift*x + self.cam_im_yshift*y +
                                       self.cam_im_zshift*z,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * -y)
        self.cam_ft_plane = CoordPlane(self.L10.position_m + self.f10*z +
                                       self.cam_ft_xshift*x + self.cam_ft_yshift*y,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * -y)

        self.sample.sample_plane = self.sample_plane
        self.sample.update()

        # Desired focal plane: aim focus here, place point source for backtrace here
        self.desired_focus_plane = copy_update(self.sample.desired_focus_plane,
            x=-x * self.backtrace_source_opening_tan_angle,
            y=y * self.backtrace_source_opening_tan_angle)


    def raytrace(self):
        """
        Forward ray tracing based on current properties.

        Output
        ------
            cam_ft_coords   S x G x 2 Vector, where S and G
                            denote the number of SLM segments and Galvo angles
                            for that corresponding dimension. Predicted camera
                            coordinates of Fourier plane camera.
            cam_im_coords   S x G x 2 Vector, where S and G
                            denote the number of SLM segments and Galvo angles
                            for that corresponding dimension.  Predicted camera
                            coordinates of Image plane camera.
        """
        # Initial Ray list
        origin, x, y, z = self.coordsystem
        self.rays = [Ray(origin, -z)]

        # Propagation to objective 1
        self.rays.append(galvo_mirror(self.rays[-1], self.galvo_plane, self.galvo_rots))
        self.rays.append(thin_lens(self.rays[-1], self.L3, self.f3))
        self.plane34_ray = self.rays[-1].intersect_plane(self.plane34)
        self.rays.append(self.plane34_ray)
        self.rays.append(thin_lens(self.rays[-1], self.L4, self.f4))
        self.rays.append(self.rays[-1].intersect_plane(self.slm_plane))
        self.slm_ray = slm_segment(self.rays[-1], self.slm_plane, self.slm_coords)
        self.rays.append(self.slm_ray)
        self.rays += abbe_lens(self.rays[-1], self.L5, self.f5)
        self.plane57_ray = self.rays[-1].intersect_plane(self.plane57)
        self.rays.append(self.plane57_ray)
        self.rays += abbe_lens(self.rays[-1], self.L7, self.f7)
        self.rays += abbe_lens(self.rays[-1], self.OBJ1, self.fobj1, n_out=self.n_water)

        # Propagate through sample
        self.rays += self.sample.raytrace(self.rays[-1])

        # Propagation from objective 2
        self.rays += [coverslip_correction(
            self.rays[-1], self.OBJ2.normal, self.obj2_coverslip_thickness_m, self.n_obj2_coverslip,
            n_out=1.0, propagation_sign=1)]
        self.rays += abbe_lens(self.rays[-1], self.OBJ2, self.fobj2, n_out=1.0)
        self.rays.append(thin_lens(self.rays[-1], self.L9, self.f9))

        # Propagation onto cameras
        self.cam_im_ray = self.rays[-1].intersect_plane(self.cam_im_plane)
        self.rays.append(self.cam_im_ray)
        self.rays.append(thin_lens(self.rays[-1], self.L10, self.f10))
        self.cam_ft_ray = self.rays[-1].intersect_plane(self.cam_ft_plane)
        self.rays.append(self.cam_ft_ray)

        # Cameras
        self.cam_ft_coords = self.cam_ft_plane.transform_rays(self.cam_ft_ray)
        self.cam_im_coords = self.cam_im_plane.transform_rays(self.cam_im_ray)

        return self.cam_ft_coords, self.cam_im_coords

    def backtrace(self):
        """
        Backward ray tracing from desired focal position to SLM.
        """
        origin, x, y, z = self.coordsystem

        # Place point source at desired focus location
        self.backrays = [point_source(self.sample.desired_focus_plane,
                         self.backtrace_Nx,
                         self.backtrace_Ny,
                         refractive_index=self.sample.n_inside)]
        self.backrays += self.sample.backtrace(self.backrays[-1])
        self.backrays += abbe_lens(self.backrays[-1], self.OBJ1, self.fobj1, n_out=1.0)
        self.backrays += [thin_lens(self.backrays[-1], self.L7, self.f7)]
        self.backrays += [thin_lens(self.backrays[-1], self.L5, self.f5)]
        self.backrays += [self.backrays[-1].intersect_plane(self.slm_plane)]
        return self.backrays[-1]

    def plot(self, ax, fraction=1, viewplane=default_viewplane()):
        """Plot the TPM setup and the current rays."""

        # Plot lenses and planes
        scale = 0.008
        plot_plane(ax, self.galvo_plane, scale, ' Galvo', viewplane=viewplane, plotkwargs={'color': 'red'})
        plot_lens(ax, self.L3, self.f3, scale, ' L3\n', viewplane=viewplane)
        plot_lens(ax, self.L4, self.f4, scale, ' L4\n', viewplane=viewplane)
        plot_plane(ax, self.slm_plane, 0.8, ' SLM', viewplane=viewplane, plotkwargs={'color': 'red'})
        plot_lens(ax, self.L5, self.f5, scale*1.7, ' L5\n', viewplane=viewplane)
        plot_lens(ax, self.L7, self.f7, scale*1.7, ' L7\n', viewplane=viewplane)

        plot_lens(ax, self.OBJ1, self.fobj1, scale*1.5, ' OBJ1\n', viewplane=viewplane)
        plot_plane(ax, self.sample_plane, scale*0.8, '', ' sample plane', viewplane=viewplane)
        self.sample.plot(ax, viewplane=viewplane, plotkwargs={'color': 'green'})
        plot_lens(ax, self.OBJ2, self.fobj2, 1.5*scale, 'OBJ2\n', viewplane=viewplane)

        plot_lens(ax, self.L9, self.f9, scale*1.2, ' L9\n', viewplane=viewplane)
        plot_lens(ax, self.L10, self.f10, scale*1.2, 'L10\n', viewplane=viewplane)

        plot_plane(ax, self.cam_ft_plane, 1000, ' Fourier Cam', viewplane=viewplane, plotkwargs={'color': 'red'})
        plot_plane(ax, self.cam_im_plane, 1000, ' Image Cam', viewplane=viewplane, plotkwargs={'color': 'red'})

        # Plot rays
        plot_rays(ax, self.rays, fraction=fraction, viewplane=viewplane, raypath_index=-1)
        ax.set_xlabel('optical axis, z (m)')
        ax.set_ylabel('x (m)')
