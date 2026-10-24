# Vector formulation and observables

The model is isotropic, nonmagnetic, local epsilon(x,z), piecewise constant along
z and periodic along x. Use exp(-i omega t), k0=2*pi/wavelength, and h=Z0 H.
The dimensionless tangential wavevectors are X=diag(kx/k0), Y=ky/k0 I.
Ky may be nonzero although the material is independent of y.

## Maxwell system

Let C be the Fourier convolution matrix of epsilon, B that of 1/epsilon,
and A=B^-1. Tangential Fourier coefficients obey

```
d/dz [Ex; Ey] = i k0 P [hx; hy]
d/dz [hx; hy] = i k0 Q [Ex; Ey]

P = [ X C^-1 Y,       I - X C^-1 X ]
    [ Y C^-1 Y - I,  -Y C^-1 X     ]

Q = [ -X Y,          X^2 - C       ]
    [ A - Y^2,       Y X           ]
```

These follow by eliminating longitudinal components from curl E=i k0 h and
curl h=-i k0 D/epsilon0. They include the coupling of transverse components.
Li's inverse factorization is applied to the x-normal displacement, Dx/epsilon0
= A Ex; the tangential constitutive relation uses C. The mode equation is
`P Q W = W diag(q^2)`, with magnetic eigenvectors `V = Q W / q`.
Homogeneous regions use explicit plane-wave modes to avoid eigenvector ambiguity
at polarization degeneracies.

The forward-mode longitudinal coefficients are

```
Ez = C^-1 (Y hx - X hy)
hz = X Ey - Y Ex
```

For a reversed mode, Ex/Ey/hz keep their sign while hx/hy/Ez reverse sign.
Select forward q with Im(q)>=0, and positive real q for propagating lossless modes.
The supported passive positive-permittivity scope avoids backward-wave materials.

At each sharp horizontal interface, match Ex, Ey, hx, hy in Fourier space. Forward
amplitudes are referenced to the layer top, backward amplitudes to the bottom;
propagation to the opposite face uses only `exp(i k0 q thickness)`. A banded
global interface system avoids growing transfer factors. Small systems use a
dense solve (at most 512 unknowns), larger systems a sparse solve. This is a
boundary-anchored global method, not a Redheffer scattering recursion.

Inverse factorization is explained in [Normal vector method for the RCWA with
automated vector field generation (2008)](https://doi.org/10.1364/OE.16.017295). The general P/Q modal construction
also appears in [VarRCWA (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9589908/).
The explicit matrices above state this implementation's conventions.

## Reconstructing physical fields

Within each slice, reconstruct continuous Dx from its Fourier series and divide
pointwise by the local epsilon to obtain Ex. Reconstruct Ey and Ez directly.
This avoids using the truncated Fourier series of a discontinuous Ex as the
physical normal field. At a discontinuity choose a documented side.

Both incident states have unit electric amplitude, so the intensity is
`|Ex|^2+|Ey|^2+|Ez|^2`. Fields are specified at y=0; the common factor
exp(i ky y) has unit modulus for our lossless incident ambient.

In normalized magnetic units, period-averaged vertical flux is
`(1/2) Re sum_orders(Ex hy* - Ey hx*)`. Physical flux includes a further 1/Z0.
The volume electric-loss density is `k0 Im(epsilon)|E|^2/(2 Z0)`; these expressions
give an independent integrated absorption check. Interface flux into an absorbing
substrate counts as transmission, even if subsequently absorbed there.

## Semi-analytic fluorescence

The scalar legacy approach is described in [Nikolaev et al., Section II.5,
arXiv:1908.11452v1](https://arxiv.org/html/1908.11452v1#S2.SS5).
Its lateral overlap/depth-integral decomposition motivates the following vector
extension; it does not supply the vector Maxwell solver itself.

In a rectangular region of a slice, combine both propagation directions into
one mode index a. Write each electric component as

```
Ec(x,z) = sum_h,a Bc[h,a] u[a] exp(i kx[h] x) exp(lambda[a]*(z-anchor[a]))
```

Forward rates are lambda=i k0 q with anchor at the top; backward rates are
lambda=-i k0 q with anchor at the bottom. Bx uses A Ex / epsilon_region,
and Bz includes the reversed sign for backward modes.

For a box of width w and center xc, the exact lateral overlap is
`L[h,g] = w*sinc((kx[h]-kx[g])*w/(2*pi))*exp(i*(kx[h]-kx[g])*xc)`.
Here sinc(t)=sin(pi*t)/(pi*t). The depth matrix is

```
D[a,b] = integral_top^bottom exp(lambda[a]*(z-anchor[a]))
                            exp(conj(lambda[b])*(z-anchor[b])) exp(-mu*z) dz
```

The relative region yield is

```
density * Re sum_a,b u[a] conj(u[b]) D[a,b]
                      * sum_c (Bc.T @ L @ conj(Bc))[a,b]
```

This preserves all forward/backward and intermode interference. Orthogonal
Cartesian components add as intensities. No real-space grid is needed.
We evaluate the depth exponential from the bounded endpoint and use expm1 plus
the finite small-exponent limit to avoid cancellation and overflow.

The output is integrated over a period and finite layers, per unit y length.
Constant atomic prefactors cancel in a normalized fixed-energy scan under
isotropic absorption. Density weights, changing geometry factors, and
angle-dependent escape attenuation do not generally cancel. The implemented
mu is only a constant exponential depth weight; arbitrary paths require a
different integration method. Semi-infinite fluorescent regions are not included.

## Accuracy and scope

Analytic box integration removes real-space quadrature error, not Fourier
truncation or slice-discretization error. Check both convergences. Use explicit
slices for resolved geometrical profiles, or the effective Fourier prescription
below when a Debye-Waller approximation is intended.

Exact zero-q diffraction cutoffs are reported as unsupported rather than
regularized silently. Very near cutoffs or grazing alpha=0 can be ill-conditioned;
alpha=0 is excluded. Results for anisotropic, magnetic, gain, spatially nonlocal,
or negative-real-permittivity media are outside this version's scope.

## Fourier roughness

`Box(..., sigma=0.0)` and `Layer.add_box(..., sigma=0.0)` accept an independent
nonnegative sigma in nm for each box. For harmonic G=2*pi*m/period, multiply the
box contrast coefficients by exp(-G^2*sigma^2/2). The m=0 coefficient is unchanged;
sigma=0 follows the original sharp-box calculation exactly. Apply damping to
both epsilon and 1/epsilon contrasts before their convolution matrices are used
in P and Q. Uniform full-period materials are unaffected.

This follows the scalar Fourier damping used in [Nikolaev et al., Section
III.1](https://arxiv.org/html/1908.11452v1#S3.SS1). Applying it separately to the
reciprocal series is this package's effective vector extension. Averaging
epsilon and averaging 1/epsilon do not yield pointwise reciprocal profiles.
Consequently this prescription is not an exact scalar smoothly graded medium
or a statistical simulation of rough surfaces, and it contains no diffuse
scattering channel. Even at very large sigma, arithmetic and harmonic averages
can differ; the solver must retain both constitutive operators.

The field reconstruction still uses nominal box material epsilon; fluorescence
uses nominal fluorescent regions without damping the lateral density overlaps.
Semi-analytic yields therefore agree with grid integration of these returned
fields over the nominal regions. For nonzero sigma that nominal-volume electric
loss integral need not equal the loss computed from the averaged modal operators
by flux balance. Do not use the sharp-box absorption/flux identity as an exact
physical assertion for this effective roughness model. Lossless modal flux
conservation and sigma=0 recovery remain validation checks.

## Interface roughness

`Layer(..., sigma=0.0)` specifies RMS height roughness in nm at the layer's top
interface. `Structure(..., substrate_sigma=0.0)` specifies roughness at the
substrate top, including a bare ambient/substrate boundary. These are independent
of `Box.sigma`. Use zero at artificial subdivisions of a physical layer.

In [Nikolaev et al., equations 12–17 and 34](https://arxiv.org/html/1908.11452v1#S2.SS3),
P is the interface field basis, Q contains propagation through a layer, and
V=P_upper^-1 P_lower matches modal amplitudes at an interface. These symbols
differ from the Maxwell P/Q operators used earlier in this document. Interface
roughness modifies V; it does not modify layer propagation or bulk eigenmodes.

For the vector basis P=[Modes.plus, Modes.minus], let a mode's signed physical
vertical wavevector be kappa=+k0*q for forward and -k0*q for backward modes.
Gaussian averaging of an interface displacement gives the elementwise rule

```
V_rough[a,b] = V[a,b] * exp(-sigma**2 * (kappa_upper[a]-kappa_lower[b])**2 / 2)
```

Use the complex wavevectors, not their absolute values, real parts, or complex
conjugates. Same-direction blocks use wavevector differences; opposite-direction
blocks use sums. The interface equation becomes
`[a_upper,b_upper] = V_rough @ [a_lower,b_lower]`, with amplitudes evaluated at
the nominal boundary. The global solve still uses top-anchored forward and
bottom-anchored backward amplitudes and only decaying thickness factors.
The zero-sigma path retains direct tangential matching without a basis inversion.

For homogeneous media this yields

```
r_rough = r_Fresnel * exp(-2*kz_upper*kz_lower*sigma**2)
t_rough = t_Fresnel * exp((kz_upper-kz_lower)**2*sigma**2 / 2)
```

These are the complex Nevot–Croce reflection/transmission conventions used in
[Haase et al. (2016), equation 3](https://doi.org/10.1107/S1600576716015776).
Here r and t refer to tangential electric amplitudes; the p-polarized admittance
is epsilon/q rather than q. The implementation retains vector coupling at
conical incidence. Applying the Gaussian rule between grating eigenmodes is
our effective modal extension, not a vector rough-surface result derived in
Nikolaev's scalar preprint. There is one width for the entire horizontal
interface, with no lateral correlation length or cross-interface correlations.

Returned fields are effective fields in nominal layers, not fields resolved
inside a graded transition. Tangential fields need not match across the nominal
rough boundary; `boundary_residual` measures the corrected matching equations.
Fluorescence still integrates total electric intensity over nominal material
regions. It does not include diffuse-field excitation or a smoothed atom-density
profile. Analytic integration and grid quadrature of these fields must agree.

This approximation does not enforce exact lossless flux balance at a rough
interface. `absorptance=1-R-T` is therefore a residual, not a separately calculated
material absorption or diffuse loss. It can even be negative for some parameters;
do not clip or interpret it as a physical diffuse-scattering prediction. The
matrix rule is intended for small roughness; use a resolved profile when large
roughness or overlapping transitions invalidate the nominal-layer picture.
Complex evanescent-mode factors can grow; numerical overflow is rejected rather
than silently clipping these factors. A successful solve alone is not evidence
that the approximation applies to a particular sample.
