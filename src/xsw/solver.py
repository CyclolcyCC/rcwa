"""Coupled Cartesian RCWA with Li factorization and boundary-anchored modes.

Convention: exp(-i omega t); h = Z0 H; dimensionless k = physical k / k0.
See docs/formulation.md for the P/Q derivation and reconstruction rules.
"""
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve

from .model import Structure, convolution, validate_order
from .constants import HC_EV_NM


@dataclass
class Modes:
    q: np.ndarray
    electric: np.ndarray  # (3, orders, modes), Ex stores Dx/epsilon0
    magnetic: np.ndarray  # (3, orders, modes), Z0 H
    plus: np.ndarray      # tangential Ex,Ey,hx,hy, for boundary matching
    minus: np.ndarray


def _branch(values):
    q = np.sqrt(np.asarray(values, complex))
    q = np.where((q.imag < 0) | ((q.imag == 0) & (q.real < 0)), -q, q)
    if np.any(np.abs(q) < 1e-10):
        raise ValueError("exact diffraction cutoff encountered; shift angle or wavelength slightly")
    return q


def _homogeneous(epsilon, kx, ky):
    """Analytic Cartesian basis avoids degenerate eigenvector ambiguity."""
    n = len(kx)
    q0 = _branch(epsilon - kx**2 - ky**2)
    q = np.tile(q0, 2)
    eye = np.eye(n, dtype=complex)
    ex, ey = np.hstack([eye, 0 * eye]), np.hstack([0 * eye, eye])
    ez = -(kx[:, None] * ex + ky * ey) / q[None, :]
    hx = ky * ez - q[None, :] * ey
    hy = q[None, :] * ex - kx[:, None] * ez
    hz = kx[:, None] * ey - ky * ex
    e = np.array([epsilon * ex, ey, ez])
    h = np.array([hx, hy, hz])
    return Modes(q, e, h, np.vstack([ex, ey, hx, hy]), np.vstack([ex, ey, -hx, -hy]))


def _patterned(eps, inv_eps, eps_x, kx, ky):
    n = len(kx)
    identity = np.eye(n, dtype=complex)
    x = np.diag(kx)
    p = np.block([[ky * x @ inv_eps, identity - x @ inv_eps @ x],
                  [ky**2 * inv_eps - identity, -ky * inv_eps @ x]])
    q_matrix = np.block([[-ky * x, x @ x - eps],
                         [eps_x - ky**2 * identity, ky * x]])
    values, w = linalg.eig(p @ q_matrix)
    q = _branch(values)
    v = (q_matrix @ w) / q[None, :]
    # Balance columns for boundary matching; modal normalization is arbitrary.
    scale = np.sqrt(np.sum(abs(w)**2 + abs(v)**2, axis=0))
    w, v = w / scale, v / scale
    ex, ey, hx, hy = w[:n], w[n:], v[:n], v[n:]
    ez = inv_eps @ (ky * hx - x @ hy)
    hz = x @ ey - ky * ex
    return Modes(q, np.array([eps_x @ ex, ey, ez]), np.array([hx, hy, hz]),
                 np.vstack([w, v]), np.vstack([w, -v]))


def _electric_coefficients(modes, epsilon):
    forward = modes.electric.copy()
    forward[0] /= epsilon
    backward = forward.copy()
    backward[2] *= -1
    return np.concatenate([forward, backward], axis=2)


def _interface_basis(upper, lower, k0, sigma):
    """Boundary equation L+ a + L- b = R+ c + R- d.

    Gaussian averaging of V=P_upper^-1 P_lower gives the complex Nevot-Croce
    prescription in the homogeneous limit. See docs/formulation.md for the
    modal extension and its limits. No layer propagation is included here.
    """
    if sigma == 0 or (np.array_equal(upper.plus, lower.plus)
                      and np.array_equal(upper.minus, lower.minus)
                      and np.array_equal(upper.q, lower.q)):
        return upper.plus, upper.minus, lower.plus, lower.minus
    p_upper = np.hstack([upper.plus, upper.minus])
    p_lower = np.hstack([lower.plus, lower.minus])
    transfer = linalg.solve(p_upper, p_lower)
    signed_upper = np.r_[upper.q, -upper.q]
    signed_lower = np.r_[lower.q, -lower.q]
    # Operate only on nonzero couplings: a zero coupling must stay zero even
    # when its evanescent-mode factor would overflow.
    rows, cols = np.nonzero(transfer)
    try:
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            delta = k0 * sigma * (signed_upper[rows] - signed_lower[cols])
            transfer[rows, cols] *= np.exp(-0.5 * delta**2)
    except FloatingPointError as error:
        raise ValueError("interface roughness factors overflow; reduce sigma or "
                         "use a resolved interface profile") from error
    size = len(upper.q)
    identity = np.eye(2 * size, dtype=complex)
    return identity[:, :size], identity[:, size:], transfer[:, :size], transfer[:, size:]


class PreparedStructure:
    """Snapshot of geometry and cached Fourier matrices for repeated solves.

    Later edits to the input Structure do not affect this object or its solutions.
    Material identity is retained so material selection continues to work.
    """

    def __init__(self, structure, max_order=15):
        if not isinstance(structure, Structure):
            raise TypeError("structure must be a Structure")
        structure.validate()
        validate_order(max_order)
        materials = [structure.ambient, structure.substrate]
        for layer in structure.layers:
            materials += [layer.background] + [b.material for b in layer.boxes]
        self.structure = deepcopy(structure, {id(m): m for m in materials})
        self.max_order = int(max_order)
        self.orders = np.arange(-max_order, max_order + 1)
        self.matrices = []
        for layer in self.structure.layers:
            eps = convolution(layer, structure.period, self.orders)
            eta = convolution(layer, structure.period, self.orders, inverse=True)
            identity = np.eye(len(self.orders), dtype=complex)
            self.matrices.append((eps, linalg.solve(eps, identity), linalg.solve(eta, identity)))

    def solve(self, *, wavelength=None, energy=None, alpha, phi=0.0, polarization="s"):
        if (wavelength is None) == (energy is None):
            raise ValueError("provide exactly one of wavelength (nm) or energy (eV)")
        value = wavelength if wavelength is not None else energy
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError("wavelength or energy must be a finite positive scalar")
        wavelength = float(value if wavelength is not None else HC_EV_NM / value)
        if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha <= np.pi / 2:
            raise ValueError("alpha must be in (0, pi/2] radians")
        if not np.isscalar(phi) or not np.isfinite(phi):
            raise ValueError("phi must be finite, in radians")
        if polarization not in ("s", "p"):
            raise ValueError("polarization must be 's' or 'p'")
        structure = self.structure
        k0 = 2 * np.pi / wavelength
        n0 = np.sqrt(structure.ambient.epsilon.real)
        kx = n0 * np.cos(alpha) * np.sin(phi) + self.orders * wavelength / structure.period
        ky = n0 * np.cos(alpha) * np.cos(phi)
        modes = [_homogeneous(structure.ambient.epsilon, kx, ky)]
        for eps, inv_eps, eps_x in self.matrices:
            uniform = np.eye(len(kx)) * eps[0, 0]
            if (np.allclose(eps, uniform, atol=1e-14, rtol=0)
                    and np.allclose(eps_x, uniform, atol=1e-14, rtol=0)):
                modes.append(_homogeneous(eps[0, 0], kx, ky))
            else:
                modes.append(_patterned(eps, inv_eps, eps_x, kx, ky))
        modes.append(_homogeneous(structure.substrate.epsilon, kx, ky))
        # s=(cos phi,-sin phi,0), p=khat cross s.
        incident_e = (np.array([np.cos(phi), -np.sin(phi), 0.0]) if polarization == "s"
                      else np.array([np.sin(alpha)*np.sin(phi),
                                     np.sin(alpha)*np.cos(phi), -np.cos(alpha)]))
        n, count = len(kx), len(structure.layers)
        size = 2 * n
        incoming = np.zeros(size, complex)
        incoming[self.max_order] = incident_e[0]
        incoming[n + self.max_order] = incident_e[1]
        phase = [np.exp(1j * k0 * mode.q * layer.thickness)
                 for mode, layer in zip(modes[1:-1], structure.layers)]
        # One block row per interface; unknowns r, (a_j,b_j)..., t.
        blocks = [[None] * (2 * count + 2) for _ in range(count + 1)]
        rhs = np.zeros((count + 1, 4 * n), complex)
        sigmas = [layer.sigma for layer in structure.layers] + [structure.substrate_sigma]
        for interface in range(count + 1):
            upper_plus, upper_minus, lower_plus, lower_minus = _interface_basis(
                modes[interface], modes[interface + 1], k0, sigmas[interface])
            if interface == 0:
                rhs[0] = -(upper_plus @ incoming)
                blocks[interface][0] = upper_minus
            else:
                blocks[interface][2 * interface - 1] = upper_plus * phase[interface-1]
                blocks[interface][2 * interface] = upper_minus
            if interface == count:
                blocks[interface][-1] = -lower_plus
            else:
                blocks[interface][2 * interface + 1] = -lower_plus
                blocks[interface][2 * interface + 2] = -lower_minus * phase[interface]
        # Profiling shows sparse assembly dominates small solves. Bound the
        # dense allocation at 512^2 complex entries (~4 MiB); use sparse beyond.
        if 4*n*(count+1) <= 512:
            zero = np.zeros((4*n, size), complex)
            matrix = np.block([[zero if b is None else b for b in row] for row in blocks])
            flat = linalg.solve(matrix, rhs.ravel())
        else:
            matrix = sparse.bmat([[None if b is None else sparse.csc_matrix(b) for b in row]
                                  for row in blocks], format="csc")
            flat = spsolve(matrix, rhs.ravel())
        coefficients = flat.reshape(2 * count + 2, size)
        if not np.isfinite(coefficients).all():
            raise ArithmeticError("nonfinite modal solution; inspect geometry and cutoff conditions")
        residual = np.linalg.norm(matrix @ coefficients.ravel() - rhs.ravel()) / max(np.linalg.norm(rhs), 1e-30)
        return Solution(self, modes, coefficients, incoming, kx, ky, wavelength,
                        float(alpha), float(phi), polarization, residual)


def prepare(structure, *, max_order=15):
    return PreparedStructure(structure, max_order)


def solve(structure, *, wavelength=None, energy=None, alpha, phi=0.0,
          polarization="s", max_order=15):
    """Solve a single incident plane wave. Angles are radians, lengths nm."""
    return prepare(structure, max_order=max_order).solve(
        wavelength=wavelength, energy=energy, alpha=alpha, phi=phi, polarization=polarization)


class Solution:
    """Reusable modal solution; field arrays have shape (len(z), len(x), 3)."""

    def __init__(self, prepared, modes, coefficients, incoming, kx, ky, wavelength,
                 alpha, phi, polarization, boundary_residual):
        self.prepared, self.structure = prepared, prepared.structure
        self.modes, self.coefficients, self.incoming = modes, coefficients, incoming
        self.kx, self.ky, self.wavelength = kx, ky, wavelength
        self.alpha, self.phi, self.polarization = alpha, phi, polarization
        self.boundary_residual = boundary_residual
        self.k0 = 2 * np.pi / wavelength
        self.orders = prepared.orders.copy()
        inc = modes[0].plus @ incoming
        self.incident_flux = self._flux(inc).sum()
        self.reflectance_orders = -self._flux(modes[0].minus @ coefficients[0]) / self.incident_flux
        self.transmittance_orders = self._flux(modes[-1].plus @ coefficients[-1]) / self.incident_flux

    @staticmethod
    def _flux(tangent):
        ex, ey, hx, hy = np.split(tangent, 4)
        return 0.5 * np.real(ex * hy.conj() - ey * hx.conj())

    @property
    def reflectance(self):
        return float(self.reflectance_orders.sum())

    @property
    def transmittance(self):
        """Flux entering the substrate at its top interface / incident flux."""
        return float(self.transmittance_orders.sum())

    @property
    def absorptance(self):
        """1-R-T; with interface roughness this is not pure material absorption.

        Rough-interface matching is an effective coherent-field approximation,
        not a flux-conserving resolved transition or a diffuse-scattering model.
        Subsequent substrate absorption is excluded in all cases.
        """
        return 1 - self.reflectance - self.transmittance

    def _amplitudes(self, region, z):
        mode = self.modes[region]
        count = len(self.structure.layers)
        if region == 0:
            a, b = self.incoming, self.coefficients[0]
            top = bottom = 0.0
        elif region == count + 1:
            a, b = self.coefficients[-1], np.zeros_like(self.incoming)
            top = bottom = self.structure.thickness
        else:
            a, b = self.coefficients[2*region-1:2*region+1]
            top, bottom = self.structure.boundaries[region-1:region+1]
        # Skip zero backward substrate amplitudes to avoid 0 * exp(overflow).
        forward = np.zeros((len(z), len(a)), complex)
        backward = np.zeros_like(forward)
        nonzero = a != 0
        forward[:, nonzero] = a[nonzero] * np.exp(1j*self.k0 * (z[:, None]-top) * mode.q[nonzero])
        nonzero = b != 0
        backward[:, nonzero] = b[nonzero] * np.exp(-1j*self.k0 * (z[:, None]-bottom) * mode.q[nonzero])
        return np.concatenate([forward, backward], axis=1)

    def fields(self, x, z, *, magnetic=False):
        """Complex E (or Z0 H), with axes (z,x,component); evaluate at y=0.

        At z interfaces return the deeper side; at x interfaces the right side.
        Ex is reconstructed from continuous Dx and local epsilon (inverse rule).
        """
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        if x.ndim != 1 or z.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError("x and z must be finite one-dimensional coordinate arrays")
        result = np.empty((len(z), len(x), 3), complex)
        regions = np.searchsorted(self.structure.boundaries, z, side="right")
        lateral = np.exp(1j * self.k0 * np.outer(self.kx, x))
        for region, mode in enumerate(self.modes):
            mask = regions == region
            if not mask.any():
                continue
            amplitudes = self._amplitudes(region, z[mask])
            if magnetic:
                backward = mode.magnetic.copy()
                backward[:2] *= -1
                vectors = np.concatenate([mode.magnetic, backward], axis=2)
            else:
                vectors = _electric_coefficients(mode, 1.0)  # reconstruct Dx first
            field = np.einsum('zm,cnm,nx->zxc', amplitudes, vectors, lateral, optimize=True)
            if not magnetic:
                if region == 0:
                    eps = self.structure.ambient.epsilon
                elif region == len(self.modes)-1:
                    eps = self.structure.substrate.epsilon
                else:
                    eps = self.structure.layers[region-1].epsilon_at(x, self.structure.period)
                field[..., 0] /= eps
            result[mask] = field
        return result

    def intensity(self, x, z):
        """Total |E|^2 / |E_inc|^2, shape (len(z), len(x)); no display scaling."""
        return np.sum(np.abs(self.fields(x, z))**2, axis=-1)

    def fluorescence(self, material=None, *, attenuation=0.0):
        """Semi-analytic relative yield in finite layers, per unit y length.

        material=None selects all materials flagged fluorescent. Explicit
        material selection works regardless of that flag. attenuation is a
        constant inverse-nm coefficient for exp(-attenuation*z), not a ray model.
        """
        from .observables import fluorescence
        return fluorescence(self, material, attenuation=attenuation)
