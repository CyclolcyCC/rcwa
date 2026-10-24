# CXRO atomic scattering factors

`sf/` contains the 92 H–U `.nff` files supplied with this project and their
original `readme.txt`, preserved byte-for-byte. They are bundled in the Python
package; runtime calculations use local resources and require no network.

Source: [CXRO atomic scattering factors](https://henke.lbl.gov/optical_constants/asf.html),
B. L. Henke, E. M. Gullikson, and J. C. Davis, Atomic Data and Nuclear Data Tables
54(2), 1993, with subsequent updates recorded in `sf/readme.txt`. These are
semi-empirical forward atomic scattering factors combining measurements and
theory, not crystallographic structure factors. Preserve the original attribution
when distributing the data. The supplied snapshot includes updates through 2023;
it is not automatically replaced with newer online tables.

## Calculation

For formula stoichiometries nu_j, molar mass M in g/mol, and density rho in
g/cm^3, the formula-unit number density in nm^-3 is
`N = rho * N_A / M * 1e-21`. The independent-atom X-ray susceptibility is

```
chi = -(r_e * wavelength_nm**2 / pi) * N * sum_j nu_j * (f1_j - 1j*f2_j)
epsilon_r = 1 + chi
```

The electron radius is converted to nm. The sign gives positive Im(chi) for
absorption under exp(-i omega t). CXRO's original readme writes its refractive
index using the opposite imaginary sign; we convert the convention explicitly.
This is the direct X-ray susceptibility relation. It corresponds to
`chi ~= -2*delta + 2j*beta`; we do not square the approximate index `1-delta+i*beta`
and introduce higher-order terms. See also the susceptibility/index relations in
[Optical constants from photoabsorption data](https://www.osti.gov/servlets/purl/792603).
Use `sqrt(1+chi)` if an index consistent with the returned dielectric function
is needed.

Molar masses are computed from the central values of the
[CIAAW 2024 abridged atomic weights](https://www.ciaaw.org/abridged-atomic-weights.htm).
An explicit `molar_mass=` in g/mol overrides this for the entire formula unit.
Elements with no standard atomic weight require this override. Isotope-specific
scattering factors and chemical-state effects are not modeled.

## Interpolation and limitations

- Interpolate f1 and f2 linearly in photon energy between neighboring samples.
  Preserve the closely spaced points bracketing absorption edges. Values between
  these points are an interpolation, not a chemical-state-resolved edge model.
- Reject out-of-range energies and interpolation brackets touching an undefined
  f1=-9999 entry. Do not extrapolate, clamp, or bridge the missing-data region.
  The usable energy range is the intersection of valid ranges of all constituents.
- Stable-sort rows in memory. For repeated energies use the last supplied row;
  this affects Er, Mg, and Pt. Do not average or modify the archived source files.
- Cache parsed tables, using importlib.resources so installed wheels work outside
  the repository. All supplied tables are included in package-data tests.
- Independent-atom additivity neglects bonding and fine structure near edges;
  use measured compound optical constants when those effects matter.
