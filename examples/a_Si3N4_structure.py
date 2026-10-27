"""Interactive fluorescence maps. Current sample: Ru on Si; edit below."""
import numpy as np
import matplotlib.pyplot as plt
import xsw

# %% Sample and scan parameters (nm; angles below in degrees)

hc = 1239.84
energy = 392.4
wavelength = hc / energy
t = 2
max_order = 12
chi_Si = xsw.chi_from_formula("Si", density=2.33, energy=energy)
chi_Si3N4 = xsw.chi_from_formula("Si3N4", density=2.8, energy=energy)
chi_SiO2 = xsw.chi_from_formula("SiO2", density=2, energy=energy)

si = xsw.Material(chi_Si, name="Si")
si3n4 = xsw.Material(chi_Si3N4, name="Si3N4", fluorescent=False)
sio2 = xsw.Material(chi_SiO2, name="SiO2", fluorescent=True)

model = xsw.Structure(period=100, substrate=si)
model.add_layer(xsw.Layer(t).add_box(-25-t, +25+t, sio2))
model.add_layer(xsw.Layer(90).add_box(-25, +25, si3n4).add_box(-25-t, -25, sio2).add_box(+25, +25+t, sio2))

alpha_deg = np.linspace(.001, 20, 200)
phi_deg = np.linspace(0, 0, 1)

# %% Calculate; maps['s'] and maps['p'] remain available in the IDE
maps = {}
for pol in ['s', 'p']:
    maps[pol] = xsw.fluorescence_map(
        model, np.deg2rad(alpha_deg)[:,None], np.deg2rad(phi_deg)[None,:],
        wavelength=wavelength, polarization=pol, max_order=max_order, material=sio2
        )

# %% Plot interactively, using a common s/p scale
plt.figure("Fluor_intensity_O")
plt.clf()
y = np.squeeze(maps["s"])
plt.plot(alpha_deg, y )
y = np.squeeze(maps["p"])
plt.plot(alpha_deg, y )
plt.show()