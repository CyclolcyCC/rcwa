"""Piecewise constant isotropic materials and one-period grating geometry."""
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True, eq=False)
class Material:
    """Constant susceptibility at the chosen wavelength (epsilon_r = 1 + chi).

    Materials have identity semantics: retain the same object when selecting a
    material for fluorescence. density is a relative fluorescent-atom weight,
    not a mass density or an automatic modification of chi.
    """

    chi: complex = 0j
    name: str = ""
    fluorescent: bool = False
    density: float = 1.0

    def __post_init__(self):
        chi = complex(self.chi)
        if not np.isfinite(chi) or chi.imag < 0 or (1 + chi).real <= 0:
            raise ValueError("chi must be finite, passive (Im >= 0), with Re(epsilon) > 0")
        if not np.isfinite(self.density) or self.density < 0:
            raise ValueError("density must be finite and nonnegative")
        object.__setattr__(self, "chi", chi)

    @property
    def epsilon(self):
        return 1 + self.chi


VACUUM = Material(name="vacuum")


def _validate_sigma(value, name="sigma"):
    if (not isinstance(value, Real) or isinstance(value, bool)
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative scalar in nm")
    return float(value)


@dataclass(frozen=True)
class Box:
    """Nominal material interval; sigma is Fourier roughness width in nm.

    Each nonzero Fourier harmonic is damped by exp(-0.5*(G*sigma)**2).
    The nominal region remains the field-reconstruction/fluorescence domain.
    """

    left: float
    right: float
    material: Material
    sigma: float = field(default=0.0, kw_only=True)

    def __post_init__(self):
        if not np.isfinite([self.left, self.right]).all() or self.left >= self.right:
            raise ValueError("box must have finite left < right")
        if not isinstance(self.material, Material):
            raise TypeError("box material must be a Material")
        object.__setattr__(self, 'sigma', _validate_sigma(self.sigma))


@dataclass
class Layer:
    """A finite slice in nm; sigma is RMS roughness of its top interface.

    Boxes replace the background and may not overlap. Interface sigma defaults
    to zero and is independent of each box's sidewall sigma.
    """

    thickness: float
    background: Material = VACUUM
    boxes: list = field(default_factory=list)
    name: str = ""
    sigma: float = field(default=0.0, kw_only=True)

    def __post_init__(self):
        self.validate()

    def validate(self, period=None):
        _validate_sigma(self.sigma)
        if not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("layer thickness must be finite and positive")
        if not isinstance(self.background, Material):
            raise TypeError("background must be a Material")
        ordered = sorted(self.boxes, key=lambda b: b.left)
        for i, box in enumerate(ordered):
            if not isinstance(box, Box):
                raise TypeError("boxes must contain Box objects")
            if i and box.left < ordered[i - 1].right:
                raise ValueError("boxes in one layer may not overlap")
            if period is not None and (box.left < -period / 2 or box.right > period / 2):
                raise ValueError("boxes must lie within [-period/2, period/2]")

    def add_box(self, left, right, material, *, sigma=0.0):
        """Add a box, optionally damping its Fourier harmonics (sigma in nm)."""
        box = Box(left, right, material, sigma=sigma)
        trial = Layer(self.thickness, self.background, self.boxes + [box], self.name,
                      sigma=self.sigma)
        self.boxes = trial.boxes
        return self

    def regions(self, period):
        """Nonoverlapping intervals, including unoccupied background."""
        self.validate(period)
        regions, left = [], -period / 2
        for box in sorted(self.boxes, key=lambda b: b.left):
            if box.left > left:
                regions.append(Box(left, box.left, self.background))
            regions.append(box)
            left = box.right
        if left < period / 2:
            regions.append(Box(left, period / 2, self.background))
        return regions

    def epsilon_at(self, x, period):
        x = (np.asarray(x) + period / 2) % period - period / 2
        eps = np.full(x.shape, self.background.epsilon, complex)
        for b in self.boxes:
            eps[(x >= b.left) & (x < b.right)] = b.material.epsilon
        return eps


@dataclass
class Structure:
    """Period in nm; z=0 at the first layer top, z increases into the sample.

    substrate_sigma is RMS roughness in nm at the substrate's top interface,
    including the ambient/substrate interface when there are no finite layers.
    """

    period: float
    substrate: Material = VACUUM
    ambient: Material = VACUUM
    layers: list = field(default_factory=list)
    substrate_sigma: float = field(default=0.0, kw_only=True)

    def validate(self):
        _validate_sigma(self.substrate_sigma, "substrate_sigma")
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError("period must be finite and positive")
        if not isinstance(self.ambient, Material) or not isinstance(self.substrate, Material):
            raise TypeError("ambient and substrate must be Materials")
        if self.ambient.chi.imag != 0:
            raise ValueError("incident ambient must be lossless")
        for layer in self.layers:
            if not isinstance(layer, Layer):
                raise TypeError("layers must contain Layer objects")
            layer.validate(self.period)

    def __post_init__(self):
        self.validate()

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("add_layer expects a Layer")
        layer.validate(self.period)
        self.layers.append(layer)
        return self

    @property
    def boundaries(self):
        return np.r_[0.0, np.cumsum([l.thickness for l in self.layers])]

    @property
    def thickness(self):
        return float(self.boundaries[-1])


def validate_order(max_order):
    if isinstance(max_order, bool) or not isinstance(max_order, Integral) or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer; orders are -M,...,M")


def convolution(layer, period, orders, inverse=False):
    """Box Fourier coefficients of epsilon or 1/epsilon, with optional damping.

    For sigma>0 apply the same Debye-Waller factor to each box's contrast
    in both series. This is a phenomenological coefficient-averaging model;
    the separately averaged series are not pointwise reciprocals.
    """
    diff = orders[:, None] - orders[None, :]
    base = 1 / layer.background.epsilon if inverse else layer.background.epsilon
    matrix = np.eye(len(orders), dtype=complex) * base
    for b in layer.boxes:
        value = 1 / b.material.epsilon if inverse else b.material.epsilon
        width = b.right - b.left
        center = (b.right + b.left) / 2
        contribution = ((value - base) * width / period * np.sinc(diff * width / period)
                        * np.exp(-2j * np.pi * diff * center / period))
        if b.sigma:
            with np.errstate(over='ignore'):
                contribution *= np.exp(-0.5 * (2*np.pi*diff/period*b.sigma)**2)
        matrix += contribution
    return matrix
