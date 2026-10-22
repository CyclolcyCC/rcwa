#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIXRF map of a Ru lamellar grating on Si (batched).
@author: kostya
"""

import mbddt as mb
import numpy as np
import matplotlib.pyplot as plt

deg = np.pi / 180
wl = 0.124

# --- materials

Si = mb.Material().set_chi(-0.9763e-5 + 0.149e-6j)
Ru = mb.Material(isfluorescent=True).set_chi((-4.4676e-5 + 2.279e-6j))
Ti = mb.Material().set_chi((-0.176e-6 + 1.004e-6j))

# --- structure
vacuum = mb.Material().set_chi(0)
sigma = 0
pitch = 136  # nm, set from your experiment
n_modes = 11  # int, set for your calculation
sym = mb.Symmetry2D(pitch, n_modes)
sample = mb.Model(sym)
sample.add_layer(mb.Layer(sym, 0).add_box(-pitch/2, pitch/2, vacuum))

sample.add_layer(mb.Layer(sym, 2.694).add_box(-10.258, 10.154, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 2.28).add_box(-16.578, 14.195, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 3.212).add_box(-18.961, 17.304, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 3.108).add_box(-22.173, 20.412, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 3.937).add_box(-23.417, 24.039, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 2.694).add_box(-5.492, 5.284, Si).add_box(-26.525, -5.492, Ti, sigma = sigma).add_box(5.284, 25.386, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 2.487).add_box(-11.916, 11.398, Si).add_box(-28.183, -12.019, Ti, sigma = sigma).add_box(11.398, 26.422, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 2.59).add_box(-15.335, 15.128, Si).add_box(-29.841, -15.335, Ti, sigma = sigma).add_box(15.128, 27.043, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 3.419).add_box(-18.029, 16.993, Si).add_box(-30.048, -18.029, Ti, sigma = sigma).add_box(16.993, 27.147, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 3.005).add_box(-18.91, 18.858, Si).add_box(-30.877, -18.91, Ti, sigma = sigma).add_box(18.858, 27.872, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 3.108).add_box(-20.257, 19.169, Si).add_box(-29.945, -20.257, Ti, sigma = sigma).add_box(19.169, 28.598, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 16.371).add_box(-19.48, 18.392, Ru, sigma = sigma).add_box(-27.872, -19.48, Ti, sigma = sigma).add_box(18.392, 28.908, Ti, sigma = sigma))
sample.add_layer(mb.Layer(sym, 12.227).add_box(-20.049, 18.858, Ru, sigma = sigma).add_box(18.858, pitch/2, Ti).add_box(-pitch/2, -20.049, Ti))
sample.add_layer(mb.Layer(sym, 4.041).add_box(-23.728, 23.106, Ru, sigma = sigma).add_box(-pitch/2, -23.728, Si).add_box(23.21, +pitch/2, Si))
sample.add_layer(mb.Layer(sym, 47.453).add_box(-pitch/2, pitch/2, Si))

#%%
AI = np.linspace(0.001, 0.7, 95)
PHI = np.linspace(0, 0.6, 45)
aa, pp = np.meshgrid(AI, PHI, indexing="ij")     # Y[i_ai, j_phi]
ai = aa.ravel() * deg
phi = pp.ravel() * deg

# Guard the entry point: the process pool re-imports this module in workers,
# so the heavy call and plotting must not run at import time.
if __name__ == "__main__":
    # Parallel evaluation across CPU cores; each worker batches its chunk and
    # pins BLAS to 1 thread to avoid oversubscription. Bit-identical to serial.
    Y = mb.calc_yield_map_parallel(sample, ai, phi, wl, Ru,
                                   chunk=1000).reshape(AI.size, PHI.size)
#%%
    plt.figure("Y")
    plt.clf()
    IMG = Y/Y.max()
    IMG = np.fliplr(IMG)
    plt.imshow(IMG,
               aspect=1.4,
               origin="lower",
               cmap="turbo",
               clim=(0.00, 1.2),
               extent=[-PHI[-1],PHI[0], AI[0], AI[-1]])
    plt.xlabel(r"Azimuth angle $\phi$, deg")
    plt.ylabel(r"Incidence angle $\alpha$, deg")
    # plt.colorbar()
    plt.show()







