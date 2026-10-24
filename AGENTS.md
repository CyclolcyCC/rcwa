# Project instructions

## Purpose and priorities

Build an efficient, easy-to-use Python package for X-ray standing wave (XSW)
calculations in periodic layered structures, including fluorescence applications
such as GIXRF/GEXRF. Support s- and p-polarized illumination with a full vectorial
electromagnetic solution within the documented geometry and material scope.

The agreed geometry is a grating periodic along x, structured along z, and
continuous (translation invariant) along y: its material distribution is
`epsilon(x, z)`. The core task is to define this structure, choose an incident
polarization basis state (s or p), and calculate the total electric intensity
`|E(x, z)|^2` on a user-specified spatial grid.

The existing code already contains most of the computational functionality. The
main task is to consolidate it into a usable, maintainable package, verify the
physics during migration, and improve performance where measurements justify it.
Prefer reusing established calculations over an unnecessary rewrite.

The package must support both ordinary installation and editable development
installation (`python -m pip install .` and `python -m pip install -e .`).

## Sources of truth

- `legacy/GIXRF_Ruthenium/a_simple_structure.py` is the primary reference for how
  users should interact with the solver.
- `legacy/GIXRF_Ruthenium/mbddt.py` contains the corresponding material, symmetry,
  layer, box, and model interface, plus batched and parallel yield calculations.
- `legacy/GIXRF_Ruthenium/a_from_tem.py` illustrates a more complex sample built
  from multiple materials and horizontal slices.
- `legacy/src/multislice/rcwa_s.py` and `rcwa_p.py` are the numerical starting
  points for the polarization solvers.
- `legacy/src/multislice/model.py` provides the existing multislice geometry and
  Fourier-matrix construction.
- `legacy/src/multislice/solver.py` contains calculation and plotting wrappers;
  extract reusable capabilities without inheriting its coupling to plotting.
- `legacy/theory/` and `legacy/documentation/` contain supporting references and
  derivations. Check them when resolving questions about the formulation.
- [Nikolaev et al., arXiv:1908.11452v1](https://arxiv.org/html/1908.11452v1),
  especially Section II.5, is the reference for the legacy semi-analytic
  fluorescence integration. The paper derives a scalar field formulation;
  the vector extension must use the actual electric-field components.

Treat `legacy/` as an archived reference. Develop the new package outside it and
preserve the archive unless the user requests changes there. New runtime code
must not depend on importing modules from `legacy/`.

## Intended user experience

Preserve the conceptual workflow of `a_simple_structure.py`:

1. Define materials, including complex susceptibility and, for fluorescence
   calculations, fluorescent identity.
2. Define periodicity and diffraction-order truncation.
3. Assemble a sample from layers containing material boxes.
4. Define wavelength or energy, incidence and azimuth, and an incident s or p
   polarization basis state.
5. Specify x and z sampling coordinates and calculate the total electric
   intensity `|Ex|^2 + |Ey|^2 + |Ez|^2` using the same interface for either input
   polarization. Document the incident-field amplitude normalization explicitly.
6. Plot or otherwise process the returned numerical results independently.

Field calculation is the primary capability and must be usable without
fluorescence inputs. Material-resolved fluorescence integration and angular
scans build on the field solution as separate operations. They need not require
sampling a dense field grid if analytical integration is available.

Define s and p relative to the incident plane and sample normal, and distinguish
these incident basis labels from Cartesian field-component names. Translational
invariance of the geometry along y must not be used to silently restrict the
incident wavevector to zero y component or remove vector-field components.

Favor a small, coherent public API with readable sample construction and
consistent behavior for s and p illumination. Chainable construction, as used in
the Ru example, is a useful reference. Exact legacy names and signatures are not
yet a compatibility requirement.

Expose numerical controls without requiring users to assemble internal solver
matrices. Document scalar and array inputs, broadcasting or scan semantics,
output shapes, normalization, and the meaning of the mode-count parameter.
Keep raw calculated values available; normalization for display must be explicit.

## Physics and numerical correctness

- Document coordinates, the depth direction and origin, angle definitions and
  units, wavelength/energy units, susceptibility and permittivity conventions,
  field normalization, and incident/reflected/transmitted wave conventions.
- The legacy interfaces disagree on angle units: the Ru example passes radians,
  while the multislice solver accepts degrees. Choose and document the new
  convention explicitly; do not silently combine these interfaces.
- Full vectorial behavior is a requirement, not a conclusion inferred from
  separate functions named s and p. Verify the formulation, field components,
  and polarization coupling for supported incidence geometries, including
  nonzero azimuth. State any restrictions explicitly.
- Distinguish complex field amplitudes, electric intensity, magnetic intensity,
  power flux, and fluorescence yield in APIs and documentation.
- In the archived implementation, `compute_field_s` returns a complex scalar
  field, while `compute_field_p` returns electric intensity. The p-polarization
  fluorescence routines integrate magnetic `|H_y|^2`. For the agreed isotropic,
  nonmagnetic optical model, fluorescence excitation uses electric intensity
  `|E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2` for both incident polarizations. Reconstruct
  the electric field when solving for magnetic amplitudes; do not substitute
  `|H|^2` or the magnitude of the Poynting vector for this observable.
- Preserve appropriate Fourier factorization, interface conditions, and outgoing
  or decaying wave branches. Do not change equations solely to simplify code or
  match a plot.
- Specify how ambient and substrate regions, material selection, smoothing,
  and fluorescence attenuation are handled. A parameter present in a legacy
  interface is not evidence that every backend implements it.
- Box roughness is `sigma=0.0` by default, in nm. Damp each box's epsilon and
  reciprocal-epsilon contrast coefficients by exp(-G^2*sigma^2/2), preserving
  the zero harmonic. This is effective Fourier averaging, not a resolved rough
  interface. Nominal material regions remain the reconstruction/integration
  domains. Document the distinction from a smooth scalar dielectric and the
  resulting limits of nominal-volume absorption/flux comparisons; see the
  Fourier roughness section of `docs/formulation.md`.
- Horizontal interface roughness uses `Layer.sigma=0.0` at the layer top and
  `Structure.substrate_sigma=0.0` at the substrate top, in nm. Keep artificial
  slice boundaries sharp. Apply Gaussian factors to V=P_upper^-1 P_lower using
  signed complex modal kz differences; Nikolaev's Q is propagation, distinct
  from the Maxwell Q operator here. Recover both complex Nevot–Croce Fresnel
  reflection and transmission amplitudes. The grating extension is an effective
  modal approximation, with nominal field/fluorescence regions and no diffuse
  field or correlation model. With interface roughness, 1-R-T is not pure
  absorption and is not guaranteed nonnegative. Validate corrected matching,
  not sharp tangential continuity or exact lossless flux conservation. See the
  interface roughness section of `docs/formulation.md`.
- Use legacy outputs for regression where applicable, together with independent
  physical checks. Agreement with legacy code alone does not establish accuracy.

## Normalized fluorescence: agreed scope

For fluorescence calculations, the default target is the shape of a normalized
angular curve or map at fixed incident photon energy for a selected element and emission line,
assuming isotropic photoabsorption. In this scope, the photoionization
cross-section is independent of incidence and azimuth angles. Its cancellation
does not rely on the angular range being small or the geometry being grazing.

Write the signal as `Y(alpha, phi) = C * J(alpha, phi)`, where `C` contains common
angle-independent factors such as the selected photoionization cross-section,
fluorescence yield, and line branching ratio. Compute the relative signal as
`J = integral n_f(r) * (|E(r)|^2 / |E_inc|^2) * W(r, alpha, phi) dV`, with `n_f`
the fluorescent-atom density and `W` any modeled escape/detection weighting.
Angle-dependent external geometry factors must also be retained when applicable.
For example, `Y / max(Y) = J / max(J)` for a nonzero signal and constant positive
`C`. Do not require atomic cross-section data to calculate this normalized shape.

Preserve spatial density/composition weights and any angle-dependent attenuation,
footprint, or detection factors; these do not generally cancel. A common uniform
density factor can be absorbed into `C`. Material optical constants are still
required to solve the fields, even when the fluorescence prefactor is omitted.

Make normalization explicit and retain the unnormalized relative signal `J`.
Use a shared scale when comparing relative s/p magnitudes; independently
normalizing each curve removes that information. Absolute yields, energy scans,
relative intensities of different elements/lines, and anisotropic or dichroic
absorption require revisiting the omitted factors rather than applying this
cancellation automatically.

Reference: the modified Sherman equation in
[Shape- and Element-Sensitive Reconstruction of Periodic Nanostructures (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8306736/)
separates energy-dependent atomic factors from the field and attenuation integral.

## Semi-analytic fluorescence integration

Preserve direct integration from modal amplitudes as the production approach for
supported layered box geometries. Field maps and integrated yields are two
outputs of the same modal solution; fluorescence must not require generating a
dense x-z map first. Grid quadrature remains useful for validation and for
weighting functions outside the supported analytic forms.

The reference separates lateral fluorescent-density overlaps from depth
integrals of modal exponentials (Section II.5, equations 41-46 in arXiv v1).
Depth-varying composition can be represented by sublayers. The legacy
`calc_yield_map` implements lateral overlaps and four forward/backward modal
products, with self-absorption omitted in that routine.

For the vector extension, integrate `|Ex|^2 + |Ey|^2 + |Ez|^2` component by
component and sum the results. Retain all mode-pair interference and the correct
component coefficients for each propagation direction. A scalar magnetic-mode
overlap cannot be reused as electric intensity without that reconstruction.
Derive this extension explicitly and check it against numerical electric-field
integration for both incident polarizations.

Use stable exponential integrals with their finite zero-exponent limits and
consistent layer reference planes. Document which density and attenuation forms
permit analytic separation; arbitrary escape-path weighting may require another
integration method. State the normalization per unit length along invariant y.

## Package design and performance

- The package and import name are `xsw`, using `pyproject.toml`, a `src/` layout,
  setuptools, and Python >=3.10. Ordinary and editable installs are supported.
- Public conventions are nm for length/wavelength, radians for angles, optional
  energy in eV, depth increasing into the sample, and unit incident electric
  amplitude for both s/p states. `max_order=M` means orders -M through M.
- CXRO tables live in `src/xsw/data/sf/`, with original bytes and attribution
  preserved. `chi_from_formula(formula, density, energy=... or wavelength=...)`
  computes static susceptibility; density here is mass density in g/cm^3,
  distinct from `Material.density`'s fluorescence weight. Include data in built
  packages and reject missing/out-of-range scattering factors. See the data
  README for the susceptibility relation and interpolation policy.
- Separate materials and geometry, numerical kernels, observables, and plotting.
  Keep plotting dependencies optional for core numerical use.
- Imports must not run simulations, create plots, start process pools, or write
  files. Guard executable examples with `if __name__ == "__main__":` when needed.
- Keep dependencies focused. NumPy and SciPy are the existing numerical basis;
  introduce additional runtime dependencies only for a concrete need.
- Validate physical inputs and numerical controls with useful error messages.
  Avoid dependence on the current working directory or `sys.path` modifications.
- Profile representative calculations before substantial optimization. Consider
  geometry-dependent precomputation, reuse, batching, and bounded-memory scans.
- Make parallel execution configurable and account for BLAS thread
  oversubscription. Compare serial, batched, and parallel results using justified
  tolerances rather than assuming bitwise identity.
- Preserve a clear serial calculation path as a reference for optimized paths.

## Validation and working approach

Migrate incrementally: establish conventions and a minimal public API, port the
Ru simple-structure example, then expand validated capabilities and optimize.
Keep proposed choices separate from decisions agreed with the user, and update
this file as the project discussion resolves them.

For numerical changes, use focused checks appropriate to the affected behavior:
homogeneous/Fresnel limits for both polarizations, electromagnetic interface
conditions, lossless energy balance where applicable, convergence with Fourier
orders and slice refinement, and analytical versus numerical observable
integration. Include checks of field components and polarization coupling when
validating the vectorial formulation.

For packaging changes, verify ordinary and editable installation in isolated
environments and imports from outside the repository. Include a small executable
example that uses the installed public API. Do not claim installation or numerical
validation passed unless it was actually run.

Keep benchmarks reproducible: record geometry, material parameters, wavelength,
angles, mode count, batch size, and execution settings. Report performance changes
alongside accuracy and memory implications.

The user has authorized implementation. The package lives in `src/xsw/`, with
examples in `examples/`, focused checks in `tests/`, and the numerical derivation
in `docs/formulation.md`. Run `python -m pytest` from an installed development
environment. `benchmarks/scan.py` records scan timings and serial/parallel
agreement. New physics capabilities require appropriate validation and updates
to the documented scope.
