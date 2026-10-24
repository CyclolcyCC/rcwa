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

Si = mb.Material().set_chi(-0.9763e-5 + 0.149e-6j)
Ru = mb.Material(isfluorescent=True).set_chi(-4.4676e-5 + 2.279e-6j)
OO = mb.Material().set_chi(0)

pitch = 136
cd = 37.1
height = 25
n_modes = 7
offset = 10
sigma = 6

sym = mb.Symmetry2D(pitch, n_modes)
sample = mb.Model(sym)
sample.add_layer(mb.Layer(sym, offset).add_box(-pitch / 2, pitch / 2, OO))
sample.add_layer(mb.Layer(sym, height).add_box(-cd / 2, cd / 2, Ru, sigma=sigma))
sample.add_layer(mb.Layer(sym, offset).add_box(-pitch / 2, pitch / 2, Si))
sample.add_layer(mb.Layer(sym, offset).add_box(-pitch / 2, pitch / 2, Si))

AI = np.linspace(0.001, 0.7, 100)
PHI = np.linspace(-0.7, 0.7, 100)
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

    # %%
    plt.figure("Y")
    plt.clf()
    plt.imshow(Y / Y.max(),
               aspect="auto",
               origin="lower",
               cmap="turbo",
               clim=(0.16, 1.15),
               extent=[PHI[0], PHI[-1], AI[0], AI[-1]])
    plt.xlabel(r"Azimuth angle $\phi$, deg")
    plt.ylabel(r"Incidence angle $\alpha$, deg")
    plt.colorbar()
    plt.show()
