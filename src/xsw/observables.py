"""Integrals of vector electric fields without a real-space sampling grid."""
import numpy as np

from .model import Material
from .solver import _electric_coefficients


def _depth_overlap(q, k0, thickness, top, attenuation):
    """Integral of boundary-anchored modal products, stable for thick layers."""
    rates = np.r_[1j * k0 * q, -1j * k0 * q]
    anchors = np.r_[np.zeros(len(q)), np.full(len(q), thickness)]
    slope = rates[:, None] + rates.conj()[None, :] - attenuation
    start = (-rates * anchors)[:, None] + (-rates.conj() * anchors)[None, :] - attenuation * top
    end = ((rates * (thickness-anchors))[:, None]
           + (rates.conj() * (thickness-anchors))[None, :] - attenuation * (top+thickness))
    # exp(start) and exp(end) are bounded. Select the endpoint that prevents
    # expm1 from evaluating an exponentially growing argument.
    from_start = slope.real <= 0
    argument = np.where(from_start, slope * thickness, -slope * thickness)
    ratio = np.ones_like(argument)
    nonzero = abs(argument) > 1e-8
    ratio[nonzero] = np.expm1(argument[nonzero]) / argument[nonzero]
    u = argument[~nonzero]
    ratio[~nonzero] = 1 + u / 2 + u**2 / 6 + u**3 / 24
    return np.exp(np.where(from_start, start, end)) * thickness * ratio


def fluorescence(solution, material=None, *, attenuation=0.0):
    if material is not None and not isinstance(material, Material):
        raise TypeError("material must be a Material or None")
    if not np.isscalar(attenuation) or not np.isfinite(attenuation) or attenuation < 0:
        raise ValueError("attenuation must be a finite nonnegative inverse-nm scalar")
    total = 0.0
    delta = solution.k0 * (solution.kx[:, None] - solution.kx[None, :])
    for index, layer in enumerate(solution.structure.layers):
        regions = [b for b in layer.regions(solution.structure.period)
                   if (b.material is material if material is not None else b.material.fluorescent)]
        if not regions:
            continue
        mode = solution.modes[index+1]
        amplitudes = solution.coefficients[2*index+1:2*index+3].ravel()
        depth = _depth_overlap(mode.q, solution.k0, layer.thickness,
                               solution.structure.boundaries[index], attenuation)
        weighted = depth * amplitudes[:, None] * amplitudes.conj()[None, :]
        for region in regions:
            width, center = region.right-region.left, (region.left+region.right)/2
            lateral = width * np.sinc(delta * width / (2*np.pi)) * np.exp(1j * delta * center)
            vectors = _electric_coefficients(mode, region.material.epsilon)
            overlap = sum(v.T @ lateral @ v.conj() for v in vectors)
            total += region.material.density * np.sum(overlap * weighted).real
    return float(total)


def normalize(values, *, scale=None):
    """Explicit normalization; supply a common positive scale for s/p comparison."""
    values = np.asarray(values, float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("values must be finite and nonempty")
    scale = float(values.max()) if scale is None else float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization scale must be finite and positive (zero signal cannot be normalized)")
    return values / scale
