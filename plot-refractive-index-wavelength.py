"""
Plot refractive index of various materials vs wavelength
"""
from torch import linspace
from materials import n_collagen_fibers, n_water, n_PBS, n_SodaLimeGlass, n_hydrated_collagen
import matplotlib.pyplot as plt

volume_fraction_collagen = 0.36

# Wavelength
wl_m = linspace(400e-9, 1100e-9, 200)
wl_nm = wl_m*1e9

# Refractive index
n_w = n_water(wl_m)
n_col = n_collagen_fibers(wl_m)
n_pbs = n_PBS(wl_m)
n_sodalime = n_SodaLimeGlass(wl_m)
n_hcol = n_hydrated_collagen(wl_m, volume_fraction_collagen)

# Plot
plt.figure(dpi=150)
plt.plot(wl_nm, n_sodalime, color='k', label='Soda Lime Glass')
plt.plot(wl_nm, n_col, color='green', label='collagen fibers')
plt.plot(wl_nm, n_hcol, color='teal',
         label=f'hydrated collagen ({volume_fraction_collagen*100:.0f}% fibers)')
plt.plot(wl_nm, n_w, color='blue', label='water')
plt.plot(wl_nm, n_pbs, color='tab:red', label='PBS')

plt.legend()
plt.xlabel('wavelength (nm)')
plt.ylabel('refractive index n')
plt.ylim((1.3, 1.65))
plt.grid(True, color=(0.93, 0.93, 0.93))
plt.title('Materials in lumen-based organ-on-a-chip')
plt.show()
