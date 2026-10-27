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
- `legacy/theory/andrle_anna.pdf` and `legacy/theory/nanomaterials-11-01647.pdf`
  define the experimental GIXRF reconstruction for Configuration B (Si3N4 grating
  on Si substrate with SiO2 oxide, 680 eV excitation, simultaneous N-Kalpha and
  O-Kalpha fluorescence).
- `legacy/self_attenuation/20200417_BestFit.npy` contains the experimental angles,
  measured N and O fluorescence intensities, JCMsuite reference curves, and
  measurement uncertainties for Configuration B at 680 eV.

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

## Fitting and experimental reconstruction: Anna Andrle Configuration B (680 eV)

Reconstruct the nanoscale profile of a periodic Si3N4 grating on a Si substrate
with native SiO2 oxide and surface carbon contamination (C2O), replicating the
experimental GIXRF reconstruction at E = 680 eV from Anna Andrle's MS thesis
(Section 4.5.1 / Figure 4.36 & Figure 4.37, and Nanomaterials 2021, 11, 1647).

### Model parameterization and nominal values
- Fixed illumination: E = 680 eV (lambda ~= 1.8233 nm), p-polarization, nominal
  pitch p = 100.834 nm, azimuth phi = 0 rad.
- Discretized trapezoidal geometry:
  - Line height `h`: nominal 93.0 nm (bounds [80, 110] nm)
  - Mid-width `w` (CD at half height): nominal 44.9 nm (bounds [35, 55] nm)
  - Sidewall angle `swa`: nominal 82.5 deg (bounds [78, 88] deg)
  - Top oxide cap thickness `t_t`: nominal 2.47 nm (bounds [1.0, 4.5] nm)
  - Groove oxide thickness `t_g`: nominal 3.67 nm (bounds [2.0, 8.0] nm)
  - Carbon contamination coating `t_C`: nominal 2.5 nm (C2O, bounds [0.5, 5.0] nm, default fixed)
  - Substrate native oxide thickness `t_sub`: 0.0 nm (excluded per agreement)
- Horizontal multislice discretization:
  - Slices the trapezoid into N horizontal `Layer` objects (default N = 10 slices).
  - Top cap layers: C2O contamination cap (thickness `t_C`), followed by SiO2 oxide cap (thickness `t_t`).
  - Upper open trench slices (height `h - t_g - t_C`):
    - Central Si3N4 box of local width w(z_j) with lateral Debye-Waller roughness `sigma_sw`
    - Conformal SiO2 oxide inner sidewalls (thickness `t_t`)
    - Conformal C2O outer contamination sidewalls (thickness `t_C`)
    - Ambient trench: vacuum
  - Groove floor contamination layer: C2O layer of thickness `t_C` coating the groove oxide
  - Lower groove slices (height `t_g`): Si3N4 core flanked by SiO2 groove oxide extending to unit cell boundaries
  - Clean Si substrate with interface roughness `sigma_vert` (native oxide excluded)
- Material assignments and optical constants at 680 eV (lambda = 1.8233 nm):
  - Substrate: Si (chi = -1.984e-3 + 3.046e-4j)
  - Grating core: Si3N4 (fluorescent for N-Kalpha, 392.4 eV; chi = -2.518e-3 + 5.166e-4j; mu_N = 2.75e-3 nm^-1)
  - Surface and groove oxides: SiO2 (fluorescent for O-Kalpha, 524.9 eV; chi = -1.746e-3 + 5.163e-4j; mu_O = 2.33e-3 nm^-1)
  - Surface contamination: C2O (rho = 1.8 g/cm^3, fluorescent for C-Kalpha, 277.0 eV; chi = -1.590e-3 + 4.537e-4j; mu_C = 6.47e-4 nm^-1)

### Physical effects in scope
- **Sidewall roughness (Debye-Waller)**: Lateral box roughness `sigma_sw` (in nm,
  nominal ~1.0-2.0 nm, bounds [0.0, 3.5] nm) damps Fourier contrast harmonics
  by exp(-0.5 * (G * sigma_sw)^2) via `Box.sigma` on grating line boxes.
- **Vertical interface roughness (Névot-Croce)**: Horizontal interface roughness
  `sigma_vert` (in nm, nominal ~0.5-1.5 nm, bounds [0.0, 2.5] nm) damps modal
  transition matrices V = P_upper^-1 P_lower by exp(-0.5 * (Delta k_z * sigma_vert)^2)
  via `Layer.sigma` at true material boundaries (top of line cap, top of groove floor,
  and `Structure.substrate_sigma`). Artificial slice-to-slice boundaries inside
  the trapezoid keep `sigma = 0.0`.
- **Incident beam divergence (<= 0.01 deg)**: The calculated angular fluorescence
  yield curves I_k(alpha) are convolved with a 1D Gaussian kernel representing beam
  angular spread sigma_div <= 0.01 deg:
  `I_conv,k(alpha) = (I_k * G_sigma_div)(alpha)`,
  where G_sigma_div(Delta alpha) = (1 / (sqrt(2*pi)*sigma_div)) * exp(-Delta alpha^2 / (2*sigma_div^2)).

### Objective function and optimization
- Experimental data is loaded from `examples/output/andrle_fig4_37_experimental_data.npz`
  (178 curated angles alpha in [0.80 deg, 14.75 deg]):
  - `exp_N`, `sigma_N`: experimental N-Kalpha counts and uncertainties
  - `exp_O`, `sigma_O`: experimental O-Kalpha counts and uncertainties
  - `exp_C`, `sigma_C`: experimental C-Kalpha counts and uncertainties
- Simultaneous weighted chi-square minimization across all measured angles and channels:
  `chi^2 = sum_{k in {N, O, C}} sum_i [(I_exp,k(alpha_i) - c_k * I_conv,k(alpha_i)) / sigma_k,i]^2`.
- Optimal elemental scaling factors `c_N, c_O, c_C`:
  Solved analytically via weighted linear least squares for any candidate geometry:
  `c_k = [sum_i (I_exp,k(alpha_i) * I_conv,k(alpha_i) / sigma_k,i^2)] / [sum_i (I_conv,k(alpha_i)^2 / sigma_k,i^2)]`,
  eliminating all scale factors from the non-linear search space.
- Residual vector for `scipy.optimize.least_squares`:
  `r_k(alpha_i) = (I_exp,k(alpha_i) - c_k * I_conv,k(alpha_i)) / sigma_k,i`.
- SciPy optimization engine: `scipy.optimize.least_squares` with TRF method and bounds.

### Implementation and architecture (`src/xsw/fit/`)
1. **Data loader (`src/xsw/fit/data.py`)**:
   Load `.npy` and `.npz` datasets, validate shapes, and expose `ExperimentData` container supporting N, O, and optional C.
2. **Discretized trapezoid builder (`src/xsw/fit/geometry.py`)**:
   Clean geometry generator `build_andrle_b_sample(params, num_slices=10)` returning
   a configured `xsw.Structure` with C2O contamination, SiO2 caps/grooves, Si3N4 core, and Si substrate.
3. **Divergence convolution (`src/xsw/fit/divergence.py`)**:
   Efficient Gaussian convolution kernel matching the angular grid.
4. **Residual and objective evaluation (`src/xsw/fit/objective.py`)**:
   Evaluate N, O, and C fluorescence yields, apply divergence convolution, compute optimal
   c_k, and return weighted residuals and chi^2 in `ForwardResult`.
5. **Optimization and persistence (`src/xsw/fit/optimizer.py`)**:
   `fit_andrle_structure` minimizes weighted chi^2 and computes parameter covariance.
   `FitResult.save()` and `FitResult.load()` provide lossless serialization to `.npz` and `.txt`.
6. **Visualization (`src/xsw/fit/visualization.py`)**:
   `plot_reconstruction_single_graph` (matching Fig. 4.37 with small markers) and `plot_reconstruction_summary`.
7. **Executable drivers (`examples/`)**:
   - `examples/fit_andrle_b.py`: 42-line minimalist optimization and visualization script.
   - `examples/plot_andrle_b.py`: 25-line instant viewer loading saved `.npz` results in < 0.05 seconds.
   - `examples/compare_gixrf_gexrf.py`: Minimalist GIXRF vs GEXRF comparison using optical reciprocity.
8. **Automated tests (`tests/test_fit.py`)**:
   Verify data loading, geometry construction, convolution conservation, analytical scaling,
   forward 3-channel evaluation, and save/load roundtrips (145 tests passing).

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
