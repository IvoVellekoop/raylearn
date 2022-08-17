"""
The Two Photon Microscope.
"""

import torch
from torch import tensor
import numpy as np
import matplotlib.pyplot as plt

from vector_functions import rotate, cartesian3d
from ray_plane import Ray, Plane, CoordPlane
from plot_functions import plot_plane, plot_lens, plot_rays
from optical import thin_lens, snells, galvo_mirror, slm_segment


# Set default tensor type to double (64 bit)
# Machine epsilon of float (32-bit) is 2^-23 = 1.19e-7
# The ray simulation contains both meter and micrometer scales,
# hence floats might not be precise enough.
# https://en.wikipedia.org/wiki/Machine_epsilon
torch.set_default_tensor_type('torch.DoubleTensor')

plt.rc('font', size=12)


class TPM():
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

        # Define refractive indices at wavelength of 715nm
        # References:
        # https://refractiveindex.info/?shelf=main&book=H2O&page=Hale
        # https://refractiveindex.info/?shelf=glass&book=soda-lime&page=Rubin-clear
        # https://refractiveindex.info/?shelf=glass&book=SCHOTT-multipurpose&page=D263TECO
        self.n_water = 1.3304
        self.n_coverslip = 1.5185

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
        self.galvo_volts_per_optical_degree = 0.5 / 1.81    # Factor of 1.81 measured on 11-05-2022
        # Galvo mirror rotation in mechanical radians per volt
        # conversion from optical scan angle -> factor of 2
        self.galvo_mech_rad_per_V = (np.pi/180) / 2 / self.galvo_volts_per_optical_degree
        self.galvo_roll = tensor((0.0,))                   # Rotation angle around optical axis

        # SLM
        # Meadowlark 1920x1152 XY Phase Series
        # https://bmpi.wiki.utwente.nl/doku.php?id=instrumentation:slm:meadowlark_slm
        # https://www.meadowlark.com/store/data_sheet/SLM%20-%201920%20x%201152%20Data%20Sheet%20021021.pdf
        # https://www.meadowlark.com/images/files/Specification%20Backgrounder%20for%20XY%20Series%20Phase%20SLMS%20-%20SB0520.pdf
        self.slm_height = 10.7e-3
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
        self.f10 = 200e-3
        self.obj1_tubelength = 200e-3           # Objective standard tubelength
        self.obj1_magnification = 16            # Objective magnification
        self.fobj1 = self.obj1_tubelength / self.obj1_magnification
        # Objective 2: Zeiss A-Plan 100x/0.8 421090-9800
        self.obj2_tubelength = 165e-3           # Objective standard tubelength
        self.obj2_magnification = 100           # Objective magnification
        self.fobj2 = self.obj2_tubelength / self.obj2_magnification

        # Lens planes transmission arm
        self.sample_zshift = tensor((0.,))
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
        self.slm_x = rotate(x * self.slm_height, z, self.slm_angle)
        self.slm_y = rotate(y * self.slm_height, z, self.slm_angle)
        self.slm_plane = CoordPlane(self.L4.position_m + (self.f4 + self.slm_zshift) * z, \
            self.slm_x, self.slm_y)

        # Lens planes to sample plane
        self.L5 = Plane(self.slm_plane.position_m + self.f5*z, -z)
        self.plane57 = CoordPlane(self.L5.position_m + self.f5 * z, x, y)
        self.L7 = Plane(self.L5.position_m + (self.f5 + self.f7)*z, -z)
        self.OBJ1 = Plane(self.L7.position_m + (self.f7 + self.fobj1)*z, -z)

        # Sample plane
        self.sample_plane = CoordPlane(self.OBJ1.position_m + self.fobj1*z +
                                       self.sample_zshift * z, -x, y)

        # Coverslip
        # Note, the 170um coverslip is ignored as it is modeled as part of the ideal lens
        # Only the 'extra thickness' is simulated
        coverslip_normal = rotate(-z, x, self.coverslip_tilt_around_x)
        coverslip_normal = rotate(coverslip_normal, y, self.coverslip_tilt_around_y)

        coverslip_extra_thickness = self.total_coverslip_thickness - 170e-6
        self.coverslip_back_plane = Plane(self.sample_plane.position_m, coverslip_normal)
        self.coverslip_front_plane = Plane(
            self.coverslip_back_plane.position_m + coverslip_extra_thickness * coverslip_normal,
            coverslip_normal)

        # Objective
        self.OBJ2 = Plane(self.sample_plane.position_m + self.fobj2*z + self.obj2_zshift*z, -z)
        self.L9 = Plane(self.OBJ2.position_m + (self.fobj2 + self.f9 + self.L9_zshift)*z, -z)
        self.L10 = Plane(self.L9.position_m + (self.f9 + self.f10)*z + self.L10_zshift*z, -z)

        # Cameras
        self.cam_im_plane = CoordPlane(self.L9.position_m + self.f9*z +
                                       self.cam_im_xshift*x + self.cam_im_yshift*y +
                                       self.cam_im_zshift*z,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * y)
        self.cam_ft_plane = CoordPlane(self.L10.position_m + self.f10*z +
                                       self.cam_ft_xshift*x + self.cam_ft_yshift*y,
                                       self.cam_pixel_size * -x,
                                       self.cam_pixel_size * -y)

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
        self.rays.append(thin_lens(self.rays[-1], self.L5, self.f5))
        self.plane57_ray = self.rays[-1].intersect_plane(self.plane57)
        self.rays.append(self.plane57_ray)
        self.rays.append(thin_lens(self.rays[-1], self.L7, self.f7))
        self.rays.append(thin_lens(self.rays[-1], self.OBJ1, self.fobj1))

        # Propagation to sample plane as ideal air lens
        self.rays.append(self.rays[-1].intersect_plane(self.sample_plane))
        self.rays.append(snells(self.rays[-1], self.sample_plane.normal, self.n_water))

        # Backpropagate through water
        self.rays.append(self.rays[-1].intersect_plane(self.coverslip_front_plane))
        self.rays.append(snells(self.rays[-1], self.coverslip_front_plane.normal, self.n_coverslip))

        # Propagate through coverslip
        self.rays.append(self.rays[-1].intersect_plane(self.coverslip_back_plane))
        self.rays.append(snells(self.rays[-1], self.coverslip_back_plane.normal, 1.))

        # # Propagation from objective 2
        self.rays.append(thin_lens(self.rays[-1], self.OBJ2, self.fobj2))
        self.rays.append(thin_lens(self.rays[-1], self.L9, self.f9))

        # Propagation onto cameras
        cam_im_ray = self.rays[-1].intersect_plane(self.cam_im_plane)
        self.rays.append(cam_im_ray)
        self.rays.append(thin_lens(self.rays[-1], self.L10, self.f10))
        cam_ft_ray = self.rays[-1].intersect_plane(self.cam_ft_plane)
        self.rays.append(cam_ft_ray)

        # Cameras
        self.cam_ft_coords = self.cam_ft_plane.transform_rays(cam_ft_ray)
        self.cam_im_coords = self.cam_im_plane.transform_rays(cam_im_ray)

        return self.cam_ft_coords, self.cam_im_coords

    def plot(self, ax=plt.gca(), fraction=1+0*  0.03):
        """Plot the TPM setup and the current rays."""

        # Plot lenses and planes
        scale = 0.008
        plot_plane(ax, self.galvo_plane, scale, ' Galvo', plotkwargs={'color': 'red'})
        plot_lens(ax, self.L3, self.f3, scale, ' L3\n')
        plot_lens(ax, self.L4, self.f4, scale, ' L4\n')
        plot_plane(ax, self.slm_plane, 0.8, ' SLM', plotkwargs={'color': 'red'})
        plot_lens(ax, self.L5, self.f5, scale, ' L5\n')
        plot_lens(ax, self.L7, self.f7, scale, ' L7\n')

        plot_lens(ax, self.OBJ1, self.fobj1, scale, ' OBJ1\n')
        plot_plane(ax, self.sample_plane, scale*0.8, '', ' sample plane')
        plot_plane(ax, self.coverslip_front_plane, scale*0.7, '', ' coverslip\n front', plotkwargs={'color': 'blue'})
        plot_plane(ax, self.coverslip_back_plane, scale*0.6, '', ' coverslip\n back', plotkwargs={'color': 'blue'})
        plot_lens(ax, self.OBJ2, self.fobj2, 0.75*scale, 'OBJ2\n')

        plot_lens(ax, self.L9, self.f9, scale, ' L9\n')
        plot_lens(ax, self.L10, self.f10, scale, 'L10\n')

        plot_plane(ax, self.cam_ft_plane, 1000, ' Fourier Cam', plotkwargs={'color': 'red'})
        plot_plane(ax, self.cam_im_plane, 1000, ' Image Cam', plotkwargs={'color': 'red'})

        # Plot rays
        plot_rays(ax, self.rays, fraction=fraction)
