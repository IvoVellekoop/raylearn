"""Example: plot a tube sample."""
import numpy as np
from matplotlib import pyplot as plt

from vector_functions import cartesian3d, rotate
from ray_plane import CoordPlane
from sample_tube import SampleTube

# Define coords and planes
origin, x, y, z = cartesian3d()
x_plane = rotate(x, z, 1)
y_plane = rotate(z, x_plane, 0.5)
viewplane = CoordPlane(origin, x_plane, y_plane)

# Define tube sample
tube = SampleTube()
tube.tube_angle = np.radians(-45)
tube.update()

# Plot tube sample
fig, ax = plt.subplots(figsize=(4, 4), dpi=110)
ax.set_aspect(1)
tube.plot(ax=ax, viewplane=viewplane)
ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.show()
