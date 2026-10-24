"""
Many-beam dynamical diffraction theory (MBDDT / RCWA) for GIXRF simulation
of 2D periodic nanostructures.

Reference:
    K. V. Nikolaev et al., J. Synchrotron Rad. (2020) 27.
    https://doi.org/10.1107/S1600577519016345

Created at PTB in 2019. @author: Konstantin V. Nikolaev
Simplified, vectorized and parallelized: 2D structures only, no structure
plotting. Every linear-algebra step is batched over a leading beam axis B, so
LAPACK is called on stacks (B, dim, dim); a flat (alpha, phi) map is evaluated
in one pass and, optionally, split across CPU cores (calc_yield_map_parallel).
The single-beam Model.calc_yield is a thin B=1 wrapper over the same kernel.
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


# ---------------------------------------------------------------------------
# Shared low-level helpers (all act on the last two axes -> batch-compatible).
# ---------------------------------------------------------------------------

def default_attenuator(*args):
    """Self-attenuation factor when there is none: U(q) -> c1*U(q) + c2."""
    return 1.0, 0.0


def block22(A, B, C, D):
    """Assemble [[A, B], [C, D]] on the last two axes into a preallocated buffer.

    Handles plain (M, M) and batched (..., M, M) blocks alike.
    """
    M = A.shape[-1]
    out = np.empty(A.shape[:-2] + (2 * M, 2 * M), dtype=complex)
    out[..., :M, :M] = A
    out[..., :M, M:] = B
    out[..., M:, :M] = C
    out[..., M:, M:] = D
    return out


def split(A):
    """Split a (batched) square matrix into its four equal blocks (last 2 axes)."""
    M = A.shape[-1] // 2
    return A[..., :M, :M], A[..., :M, M:], A[..., M:, :M], A[..., M:, M:]


def _U_box(q, a, b):
    """Fourier transform of a box indicator on [a, b], evaluated at q.

    Vectorized; returns (b - a) where q == 0.
    """
    q = np.asarray(q, dtype=complex)
    safe = np.where(q == 0, 1.0, q)
    return np.where(q == 0, b - a,
                    1j / safe * (np.exp(-1j * q * b) - np.exp(-1j * q * a)))


def _U_safe(q, expr_func, q0):
    """Evaluate expr_func(q) elementwise, substituting q0 where q == 0."""
    q = np.asarray(q, dtype=complex)
    safe = np.where(q == 0, 1.0, q)
    return np.where(q == 0, q0, expr_func(safe))


def get_toeplitz(chi_hat):
    """Circulant Toeplitz matrix C of Fourier components, eq. (11)."""
    vals = list(chi_hat.values())
    mid = len(vals) // 2
    return linalg.toeplitz(vals[mid:], np.flipud(vals[:mid + 1]))


def angle2wavevect(ai, phi, wl):
    """Wave vector(s) in vacuum from incidence ai, azimuth phi, wavelength wl.

    Scalars give a (3,) vector; equal-length arrays give a (B, 3) stack.
    """
    k0 = 2 * np.pi / wl
    K = k0 * np.stack([np.cos(ai) * np.sin(phi),
                       np.cos(ai) * np.cos(phi),
                       -np.sin(ai)], axis=-1)
    return K


# ---------------------------------------------------------------------------
# Geometry / material model.
# ---------------------------------------------------------------------------

class Beam:
    """Container for incident-beam parameters (K, components, wl, angles)."""

    def __init__(self, ai, phi, wl):
        self.K = self.kx, self.ky, self.kz = angle2wavevect(ai, phi, wl)
        self.wl = wl
        self.alpha = ai
        self.phi = phi


class Symmetry2D:
    """1D elementary cell of a 2D structure (periodic in x, finite depth in z).

    nodes : list of (hx, hy) reciprocal-space coordinates
    index : list of (mx, my) diffraction indices
    """

    def __init__(self, period, n_modes):
        self.cell = (period, period)
        self.n_modes_x = 2 * n_modes
        self.n_modes_y = 0
        Mx, My = self.n_modes_x, self.n_modes_y
        Dx, Dy = self.cell
        self.nodes, self.index = [], []
        for mx in range(-Mx, Mx + 1):
            for my in range(-My, My + 1):
                self.nodes.append((2 * np.pi * mx / Dx, 2 * np.pi * my / Dy))
                self.index.append((mx, my))


class Material:
    """Material optical properties (constant complex chi) and visualization tags."""

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    def __init__(self, tag="", color=None, alpha=0.5, isfluorescent=False):
        self._isfluorescent = isfluorescent
        self._tag = tag
        self.color = color or next(self.colors)
        self.alpha = alpha

    def set_chi(self, chi):
        """Set the (complex, constant) dielectric susceptibility chi."""
        self.chi = complex(chi)
        return self


class Box:
    """Rectangular box primitive in the (x, y) plane."""

    def __init__(self, left, right, bck, fwd, sigma, material,
                 attenuator=default_attenuator):
        self.left, self.right = left, right
        self.bck, self.fwd = bck, fwd
        self.sigma = sigma
        self.material = material
        self.attenuator = attenuator

    def get_image(self, chi_a, unit_cell, nodes):
        """Fourier components of chi for this box on the node grid, eq. (4)."""
        Dx, Dy = unit_cell
        h = np.array(nodes)
        hx, hy = h[:, 0], h[:, 1]
        damp = np.exp(-(hx**2 + hy**2) * self.sigma**2 / 2)
        return chi_a / Dx / Dy * _U_box(hx, self.left, self.right) \
            * _U_box(hy, self.bck, self.fwd) * damp

    def get_PSI(self, H, Mx, My):
        """Lateral overlap matrix Phi, eq. (30). Vectorized over node pairs."""
        H = np.array(H)
        hx, hy = H[:, 0], H[:, 1]
        psi_x = _U_box(hx[:, None] - np.conj(hx[None, :]), self.left, self.right) if Mx else 1
        psi_y = _U_box(hy[:, None] - np.conj(hy[None, :]), self.bck, self.fwd) if My else 1
        return psi_x * psi_y


class Layer:
    """Container for boxes sharing one symmetry and thickness."""

    def __init__(self, sym, thickness, tag=''):
        self._sym = sym
        self._thickness = thickness
        self._materials = set()
        self._up = self._dwn = None
        self.tag = tag
        self.primitives = {}

    def add_box(self, left, right, material, sigma=0, attenuator=default_attenuator):
        """Add a box spanning the full cell in y (2D grating line)."""
        Dx, Dy = self._sym.cell
        box = Box(left, right, -Dy / 2, Dy / 2, sigma, material, attenuator)
        self.primitives.setdefault(material, []).append(box)
        self._materials.add(material)
        return self

    def get_image(self, wl=None):
        """Dictionary {(hx, hy): chi_h} of total Fourier components, eq. (3).

        chi is a constant complex scalar, so wl is unused (kept for symmetry).
        """
        nodes = self._sym.nodes
        chi_hat = np.zeros(len(nodes), dtype=complex)
        for material in self._materials:
            for prim in self.primitives.get(material, []):
                chi_hat += prim.get_image(material.chi, self._sym.cell, nodes)
        return dict(zip(nodes, chi_hat))


# ---------------------------------------------------------------------------
# Batched physics kernel (B = number of beams along the leading axis).
# ---------------------------------------------------------------------------

def _layer_PQ(layer, K, wl):
    """Batched (P, Q, E, kgz) for one layer over all beams.

    K : (B, 3) wave vectors. The Toeplitz part of A is beam-independent; only
    the diagonal (spherical dispersion, eq. 6) varies per beam.
    """
    B = K.shape[0]
    chi_hat = layer.get_image(wl)
    nodes = np.array(layer._sym.nodes)
    n = len(nodes)
    sub = nodes[n // 4:3 * n // 4 + 1]                  # (M, 2)
    M = sub.shape[0]

    C = (2 * np.pi / wl)**2 * get_toeplitz(chi_hat)     # (M, M), shared
    A = np.broadcast_to(C, (B, M, M)).copy()            # (B, M, M)

    kap = (K[:, 0]**2 + K[:, 1]**2 + K[:, 2]**2) * (1 + chi_hat[(0, 0)])
    diag = kap[:, None] - ((K[:, 0, None] + sub[None, :, 0])**2
                           + (K[:, 1, None] + sub[None, :, 1])**2)   # (B, M)
    idx = np.arange(M)
    A[:, idx, idx] = diag

    k2, E = np.linalg.eig(A)                            # (B, M), (B, M, M)
    kgz = np.sqrt(k2.astype(complex))                   # (B, M)

    O = np.zeros((B, M, M), dtype=complex)
    I = np.broadcast_to(np.eye(M) + 0j, (B, M, M))
    k_hat = np.zeros((B, M, M), dtype=complex)
    k_hat[:, idx, idx] = kgz
    P = block22(E, O, O, E) @ block22(I, I, k_hat, -k_hat)
    qm = np.zeros((B, M, M), dtype=complex)
    qm[:, idx, idx] = np.exp(1j * kgz * layer._thickness)
    Q = block22(O, O, O, qm)
    return P, Q, E, kgz


def _calc_M(Pa, Pb, Q):
    """Stable interface transfer matrix, eq. (20). Batched."""
    X11, X12, X21, X22 = split(np.linalg.inv(Pa) @ Pb)
    Qm = split(Q)[3]
    X11_ = np.linalg.inv(X11)
    M11 = Qm @ X11_
    return block22(M11, -M11 @ X12 @ Qm,
                   X21 @ X11_, (X22 - X21 @ X11_ @ X12) @ Qm)


def _calc_Wb(Wa, M):
    """Recursive accumulation of W matrices, eqs. (19)-(24). Batched."""
    Wa11, Wa12, Wa21, Wa22 = split(Wa)
    M11, M12, M21, M22 = split(M)
    I = np.broadcast_to(np.eye(M11.shape[-1]) + 0j, M11.shape)
    A = M11 @ np.linalg.inv(I - Wa12 @ M21)
    Bb = Wa22 @ np.linalg.inv(I - M21 @ Wa12)
    return block22(A @ Wa11, M12 + A @ Wa12 @ M22,
                   Wa21 + Bb @ M21 @ Wa11, Bb @ M22)


def _calc_RT(M, W, R, T0):
    """Recursive reflection/transmission amplitudes. Batched."""
    M11, M12, M21, M22 = split(M)
    W11, W12, W21, W22 = split(W)
    I = np.broadcast_to(np.eye(M11.shape[-1]) + 0j, M11.shape)
    Rn = np.linalg.inv(I - M21 @ W12) @ (M22 @ R + M21 @ W11 @ T0)
    return Rn, W11 @ T0 + W12 @ Rn


def _calc_amp(stack, PQ):
    """Batched plane-wave amplitudes. Returns AA (B, L, 2M), AA_ (B, L, 2M)."""
    L = len(stack)
    MM = [_calc_M(PQ[0][0], PQ[1][0], PQ[1][1])]
    WW = [MM[-1]]
    for i in range(1, L - 1):
        MM.append(_calc_M(PQ[i][0], PQ[i + 1][0], PQ[i + 1][1]))
        WW.append(_calc_Wb(WW[-1], MM[-1]))

    B = MM[0].shape[0]
    m = MM[0].shape[-1] // 2
    T0 = np.zeros((B, m, 1), dtype=complex)
    T0[:, m // 2, 0] = 1.0

    RR = [np.zeros((B, m, 1), dtype=complex)]
    TT = [split(WW[-1])[0] @ T0]
    for i in range(len(WW) - 1, 0, -1):
        Ri, Ti = _calc_RT(MM[i], WW[i - 1], RR[-1], T0)
        RR.append(Ri)
        TT.append(Ti)
    RR.append(split(WW[-1])[2] @ T0)
    TT.append(T0)
    RR, TT = RR[::-1], TT[::-1]

    TT = np.concatenate(TT, axis=2)        # (B, m, L)
    RR = np.concatenate(RR, axis=2)        # (B, m, L)
    AA = np.concatenate([TT, RR], axis=1).transpose(0, 2, 1)   # (B, L, 2m)

    AA_ = [(PQ[0][1] @ AA[:, 0][..., None])[..., 0]]
    for i in range(L - 1):
        AA_.append((np.linalg.inv(PQ[i + 1][0]) @ PQ[i][0] @ AA[:, i][..., None])[..., 0])
    AA_ = np.stack(AA_, axis=1)            # (B, L, 2m)
    return AA, AA_


def calc_yield_map(model, ai, phi, wl, material):
    """Fluorescence yield for a flat list of beams (batched).

    ai, phi  : 1D arrays of equal length B (radians); scalars also accepted.
    material : the fluorescent Material to integrate over.
    Returns Y : 1D array of length B.
    """
    ai = np.atleast_1d(ai).astype(float)
    phi = np.atleast_1d(phi).astype(float)
    K = angle2wavevect(ai, phi, wl)            # (B, 3)
    B = K.shape[0]

    stack = model._stack
    PQ = [_layer_PQ(layer, K, wl) for layer in stack]
    AA, AA_ = _calc_amp(stack, PQ)
    M = AA_.shape[2] // 2

    Mx, My = model._sym.n_modes_x, model._sym.n_modes_y
    H = model._sym.nodes
    H = np.array(H[len(H) // 4:3 * len(H) // 4 + 1])

    Y = np.zeros(B)
    for idx, layer in enumerate(stack):
        if material not in layer._materials:
            continue
        t = layer._thickness
        T = AA_[:, idx, :M]            # (B, M)
        R = AA[:, idx, M:]             # (B, M)
        _, _, E, kgz = PQ[idx]         # E:(B,M,M)  kgz:(B,M)

        km = kgz[:, :, None]
        kn = np.conj(kgz)[:, None, :]
        Tm, Tn = T[:, :, None], np.conj(T)[:, None, :]
        Rm, Rn = R[:, :, None], np.conj(R)[:, None, :]

        # Vertical overlap integrals U(q), eqs. (31)-(32) (self-absorption omitted).
        Utt = _U_safe(-km + kn, lambda q: -1j / q * (1 - np.exp(-1j * q * t)), t)
        Utr = _U_safe(-km - kn, lambda q: -1j / q * (np.exp(-1j * kn * t) - np.exp(1j * km * t)), t)
        Urt = _U_safe(km + kn, lambda q: -1j / q * (np.exp(1j * km * t) - np.exp(-1j * kn * t)), t)
        Urr = _U_safe(km - kn, lambda q: -1j / q * (np.exp(1j * q * t) - 1), t)
        W = Tm * Tn * Utt + Tm * Rn * Utr + Rm * Tn * Urt + Rm * Rn * Urr   # (B, M, M)

        for prim in layer.primitives.get(material, []):
            hx = H[:, 0]
            psi_x = _U_box(hx[:, None] - np.conj(hx[None, :]), prim.left, prim.right) if Mx else 1
            PSI = psi_x                                                 # (M, M), beam-independent
            F = E @ W @ np.conj(E.transpose(0, 2, 1))                   # (B, M, M)
            Y += np.sum(PSI[None] * F, axis=(1, 2)).real
    return Y


# ---------------------------------------------------------------------------
# Parallel map evaluation across CPU cores (process pool).
# Points are independent, so we split them into chunks and run chunks in
# parallel. Each worker pins BLAS to 1 thread to avoid oversubscription
# (processes x BLAS-threads), then uses the batched kernel internally.
# ---------------------------------------------------------------------------

_WORKER = {}


def _find_material(model, key):
    """Resolve the fluorescent material inside a (possibly unpickled) model.

    `model._materials` is a set; its iteration order is NOT stable across
    processes (hash randomization), so a positional index is unsafe. Match on
    stable attributes: the tag if unique, otherwise the single fluorescent flag.
    """
    tag, isfl = key
    cands = [m for m in model._materials if m._tag == tag] if tag else []
    if len(cands) == 1:
        return cands[0]
    cands = [m for m in model._materials if m._isfluorescent == isfl]
    if len(cands) == 1:
        return cands[0]
    raise ValueError("cannot uniquely resolve material across processes; "
                     "give the fluorescent material a unique tag")


def _worker_init(model_blob, wl, n_blas):
    """Runs once per worker process: pin BLAS threads and unpickle the model."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_blas)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n_blas)
    except Exception:
        pass
    _WORKER["model"] = pickle.loads(model_blob)
    _WORKER["wl"] = wl


def _worker_chunk(args):
    """Compute one chunk; material is referenced by stable attributes."""
    ai, phi, key = args
    model = _WORKER["model"]
    return calc_yield_map(model, ai, phi, _WORKER["wl"], _find_material(model, key))


def _seq_chunk(model, task, wl):
    ai, phi, key = task
    return calc_yield_map(model, ai, phi, wl, _find_material(model, key))


def calc_yield_map_parallel(model, ai, phi, wl, material,
                            n_workers=None, chunk=512, n_blas=1):
    """Parallel GIXRF yield over a flat list of beams.

    ai, phi   : 1D arrays of angles (radians), length B.
    n_workers : number of processes (default: os.cpu_count()).
    chunk     : points per task (batched internally by calc_yield_map).
    n_blas    : BLAS threads per worker (keep 1 unless n_workers < cores).

    Returns Y : 1D array, length B. Bit-identical to the serial calc_yield_map.
    """
    ai = np.atleast_1d(ai).astype(float)
    phi = np.atleast_1d(phi).astype(float)
    n = ai.size
    if n_workers is None:
        n_workers = os.cpu_count()

    key = (material._tag, material._isfluorescent)   # stable across processes
    tasks = [(ai[k:k + chunk], phi[k:k + chunk], key)
             for k in range(0, n, chunk)]

    # Small problems: skip process overhead.
    if n_workers == 1 or len(tasks) == 1:
        return np.concatenate([_seq_chunk(model, t, wl) for t in tasks])

    blob = pickle.dumps(model)
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init,
                             initargs=(blob, wl, n_blas)) as ex:
        parts = list(ex.map(_worker_chunk, tasks))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Layer stack + convenience single-beam API (B=1 wrapper over the kernel).
# ---------------------------------------------------------------------------

class Model:
    """Layer stack plus the simulation logic (fluorescence yield)."""

    def __init__(self, sym, attenuation=False):
        self._sym = sym
        self._stack = []
        self._coursor = 0
        self._materials = set()
        self._attenuation = attenuation

    def __iter__(self):
        return iter(self._stack)

    def add_layer(self, layer):
        if self._sym != layer._sym:
            raise ValueError("Can not add layer. Wrong symmetry")
        layer = copy(layer)
        layer._up = self._coursor
        self._coursor -= layer._thickness
        layer._dwn = self._coursor
        self._stack.append(layer)
        self._materials |= layer._materials
        return self

    def calc_yield(self, beam):
        """Fluorescent intensity per material for a single beam.

        Thin B=1 wrapper over the batched kernel; returns {material: intensity}
        for every fluorescent material in the stack.
        """
        ai = np.atleast_1d(beam.alpha).astype(float)
        phi = np.atleast_1d(beam.phi).astype(float)
        Y = {}
        for material in self._materials:
            if material._isfluorescent:
                Y[material] = calc_yield_map(self, ai, phi, beam.wl, material)[0]
        return Y
