# xsw

Vector X-ray standing waves in isotropic line gratings: periodic in **x**, layered
in **z**, and invariant along **y**. Calculate complex electric fields, total
electric intensity, reflection/transmission, and relative fluorescence directly
from modal amplitudes. Both incident s and p states use the coupled Maxwell system.

## Install

Requires Python 3.10 or newer. From a checkout, preferably in a virtual environment:

```bash
python -m pip install .
# For development, with tests and optional plotting:
python -m pip install -e '.[test,plot]'
python -m pytest
```

Runtime dependencies are NumPy, SciPy, and threadpoolctl. Plotting is optional.

## Define a sample and solve

```python
import numpy as np
import xsw

si = xsw.Material(chi=-0.9763e-5 + 0.149e-6j, name="Si")
ru = xsw.Material(chi=-4.4676e-5 + 2.279e-6j, name="Ru", fluorescent=True)

sample = xsw.Structure(period=136, substrate=si)
sample.add_layer(xsw.Layer(thickness=25).add_box(-18.55, 18.55, ru))

solution = xsw.solve(
    sample, wavelength=0.124,
    alpha=np.deg2rad(0.3), phi=np.deg2rad(0.2),
    polarization="p", max_order=15,
)
x = np.linspace(-68, 68, 301)
z = np.linspace(-10, 45, 251)
electric = solution.fields(x, z)      # complex array (251, 301, 3): Ex, Ey, Ez
intensity = solution.intensity(x, z)  # real array (251, 301): sum |Ec|^2
yield_ru = solution.fluorescence(ru)  # semi-analytic; does not use x or z
print(solution.reflectance, solution.transmittance)
```

Keep each `Material` object and reuse it in layers and material selection. Equal
optical constants or equal names do not merge separate material objects.
`fluorescence()` with no selection sums materials marked `fluorescent=True`;
an explicit material selects it regardless of the flag.

For repeated illumination settings, cache the geometry preparation:

```python
prepared = xsw.prepare(sample, max_order=15)
solution_s = prepared.solve(wavelength=0.124, alpha=0.005, phi=0.002, polarization="s")
solution_p = prepared.solve(wavelength=0.124, alpha=0.005, phi=0.002, polarization="p")
```

Preparation snapshots the input geometry and caches Fourier matrices and their
inverses. Later edits to `sample` do not alter an existing solution. Treat prepared
objects, solutions, and their exposed arrays as read-only. Re-prepare to change geometry.

## Susceptibility from formula and density

The package includes the supplied CXRO elemental scattering-factor tables:

```python
chi = xsw.chi_from_formula("SiO2", density=2.2, energy=10000)  # g/cm^3, eV
silica = xsw.Material(chi=chi, name="SiO2")

# Equivalent wavelength input (nm); energies/wavelengths can also be arrays.
chi = xsw.chi_from_formula("SiO2", density=2.2, wavelength=0.124)
mass = xsw.molar_mass("SiO2")  # 60.083 g/mol
```

Specify exactly one of `energy` or `wavelength`. Formula parsing supports
`Si3N4`, `Ca(OH)2`, nested parentheses, and fractional counts such as
`Si0.7Ge0.3`. Symbols are case-sensitive. Hydrate-dot, isotope, and charge
notation are unsupported. An optional `molar_mass=` overrides the computed
formula mass in g/mol; it is required for elements without standard atomic weights.

This function's `density` is **mass density in g/cm^3**, whereas `Material.density`
remains the independent relative fluorescence weight. Calculated chi is fixed at
the requested energy: recreate the material when changing energy. Tables are
loaded locally, interpolated linearly in energy, and never extrapolated. Missing
f1 values are rejected. See [data provenance and conventions](src/xsw/data/README.md).

## Conventions

- All lengths and wavelengths are **nm**, angles are **radians**. Alternatively
  supply `energy=...` in eV, instead of `wavelength`. Material susceptibility must
  already correspond to that energy. The solver does not automatically update
  chi; use `chi_from_formula` explicitly for a new energy if needed.
- The first layer starts at z=0; z increases downward. Ambient occupies z<0 and
  the substrate occupies z>=total thickness. Both are homogeneous half-spaces.
- The incident wavevector is
  `k/k0 = n_ambient * (cos(alpha)*sin(phi), cos(alpha)*cos(phi), sin(alpha))`.
  Thus phi=0 means incidence along the grating lines. `0 < alpha <= pi/2`.
- With `exp(-i*omega*t)`, passive susceptibility has positive imaginary part.
  The supported initial scope is scalar, nonmagnetic, passive permittivity with
  positive real part, and a lossless incident ambient.
- `s = (cos(phi), -sin(phi), 0)` and `p = khat cross s`. At normal incidence phi
  specifies the limiting polarization basis. Both have unit incident electric
  amplitude. All internal components are retained for either incident state.
- `max_order=M` includes **2M+1 spatial diffraction orders**, -M through M, with
  two vector eigenmodes per order. Check convergence for each sample/illumination.
- Fields are evaluated at y=0. For the real incident ky used here, other y planes
  differ by a common phase and have the same intensity.
- `fields(..., magnetic=True)` returns **Z0 H**, not SI H. Divide by vacuum
  impedance to obtain H in units consistent with the chosen electric amplitude.
- Interface samples use the deeper side in z and the right side in x. The
  normal electric field is reconstructed from Dx/local epsilon to respect the
  lateral discontinuity. Use one-sided samples when integrating material boxes.
- `reflectance_orders` and `transmittance_orders` follow `solution.orders` and
  sum both output polarizations. `transmittance` is power entering the substrate
  at its top; `absorptance = 1-R-T` excludes subsequent substrate absorption.
  With interface roughness, this remainder is not pure material absorption.

Boxes replace the layer background (vacuum by default), must not overlap, and
must lie inside [-period/2, period/2]. Use `Layer(thickness, background=material)`
for a homogeneous film. Approximate trapezoids or measured profiles with slices.
Zero-thickness layers and exact diffraction cutoffs are rejected explicitly.

Optional sidewall roughness uses a Debye-Waller factor on each box's Fourier
contribution. `sigma` is a nonnegative width in nm, defaulting to **0.0**:

```python
layer = xsw.Layer(90).add_box(-25, 25, material, sigma=2.0)
# Equivalently:
box = xsw.Box(-25, 25, material, sigma=2.0)
```

The factor is `exp(-0.5*(2*pi*m*sigma/period)**2)`, where m is the Fourier
harmonic index (the difference of diffraction orders in a convolution matrix).
It preserves the zeroth coefficient and damps both epsilon and reciprocal-epsilon
box contrasts. Different boxes can have different sigma values. This is an
effective Fourier roughness prescription; it does not simulate random surfaces
or diffuse scattering. Field reconstruction and fluorescence integration use
the nominal box intervals. See [roughness conventions](docs/formulation.md#fourier-roughness).

Horizontal interface roughness uses Nevot–Croce factors. Each layer's `sigma`
applies to its **top** interface; `substrate_sigma` applies to the final interface:

```python
sample = xsw.Structure(100, substrate=si, substrate_sigma=0.8)
sample.add_layer(xsw.Layer(90, sigma=1.0).add_box(-25, 25, material, sigma=2.0))
# Top interface: 1 nm; sidewalls: 2 nm; substrate interface: 0.8 nm.
```

Both interface parameters default to **0.0 nm** and must be finite and
nonnegative. With no finite layers, `substrate_sigma` describes the
ambient/substrate interface. Leave artificial slicing boundaries at zero.
The correction acts on interface matching, preserving propagation within each
slice. It recovers the complex Fresnel amplitude factor
`exp(-2*kz_upper*kz_lower*sigma**2)` for a homogeneous interface, for both s and p.
For gratings this is an effective modal extension; it does not resolve surface
profiles or diffuse scattering. Fields and fluorescence use the nominal layer
boundaries. See [interface roughness](docs/formulation.md#interface-roughness)
for the transmission convention and limitations of flux interpretation.

## Fluorescence and scans

The base excitation observable is the total electric intensity for both input
polarizations. The returned yield is a relative integral per unit length along y,
over one period and the finite layers. It includes all mode-pair interference.
No photoionization cross-section is needed for a normalized fixed-energy angular
curve for one line under the isotropic assumption. See [formulation](docs/formulation.md).

`Material.density` is a nonnegative relative fluorescent-atom weight (default 1).
It does not change chi. `fluorescence(material, attenuation=mu)` optionally weights
the integral by `exp(-mu*z)` with mu in nm^-1. This is a simple depth-attenuation
model, not geometry-dependent escape-ray tracing. Ambient/substrate fluorescence,
absolute atomic yields, footprint corrections, and detector response are not
included automatically.

```python
alpha = np.deg2rad(np.linspace(0.02, 0.7, 30))
phi = np.deg2rad(np.linspace(-0.7, 0.7, 40))
yields = xsw.fluorescence_map(
    sample, alpha[:, None], phi[None, :],
    wavelength=0.124, polarization="p", max_order=15, material=ru,
    workers=1, chunk_size=32, blas_threads=1,
)
normalized = xsw.normalize(yields)
```

Angles broadcast using NumPy rules: the example returns `(30,40)`; equal-length
1D arrays describe paired angles. Use a `__main__` guard for `workers>1`.
Scans reuse geometry, submit a bounded number of chunks, and limit BLAS threads.
They solve each illumination separately; eigenproblems are not stacked in memory.
Floating-point agreement between serial and parallel runs is tolerance-based.
Use a common `scale=` with `normalize` to preserve relative s/p magnitudes.

## Examples and validation

```bash
python examples/a_simple_structure.py --output examples/output/ru.npz
python examples/field_maps.py --output examples/output/fields.png
python benchmarks/scan.py --output benchmarks/latest.json
```

The test suite covers vacuum vector fields, s/p Fresnel interfaces and films,
lossless grating energy balance, tangential interface equations, Maxwell curls,
absorption/flux consistency, thick-layer stability, Fourier/slice convergence,
analytic versus numerical fluorescence, and serial/parallel scan consistency.
Rough-interface checks include complex Nevot–Croce Fresnel amplitudes,
multilayer transfer-matrix results, and nominal-region fluorescence quadrature.
These checks validate the implementation against analytical limits and internal
physical identities; comparison with an independent full-vector grating solver
remains a useful additional benchmark. Tests use modest examples; users must
establish convergence for their own parameter ranges.

The numerical method and its limitations are documented in
[docs/formulation.md](docs/formulation.md). Archived scripts and papers remain
in `legacy/`; the installed package has no runtime dependency on that directory.
