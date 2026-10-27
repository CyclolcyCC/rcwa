"""Compound susceptibility from bundled CXRO atomic scattering factors.

Independent-atom X-ray relation, exp(-i omega t):
chi = -r_e lambda^2/pi * (rho N_A/M) * sum_j nu_j (f1_j - i f2_j).
Density is g/cm^3, formula molar mass is g/mol, wavelength is nm.
See data/README.md for provenance, interpolation policy, and scope.
"""
from collections import defaultdict
from functools import lru_cache
from importlib.resources import files
import re
from typing import overload

import numpy as np
from scipy.constants import Avogadro, physical_constants

from .constants import HC_EV_NM

# Central values of CIAAW's abridged standard atomic weights (2024), H through U.
# None means no standard atomic weight: require an explicit formula molar mass.
# https://www.ciaaw.org/abridged-atomic-weights.htm
_SYMBOLS = """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn
Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb
Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir
Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U""".split()
_WEIGHTS = [
    1.0080, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180,
    22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.95, 39.098, 40.078,
    44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,
    69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.222,
    92.906, 95.95, None, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71,
    121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24,
    None, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05,
    174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59,
    204.38, 207.2, 208.98, None, None, None, None, None, None, 232.04, 231.04, 238.03,
]
_ATOMIC_WEIGHTS = dict(zip(_SYMBOLS, _WEIGHTS))
_NUMBER = re.compile(r'(?:\d+(?:\.\d+)?|\.\d+)')
_ELEMENT = re.compile(r'[A-Z][a-z]?')


def _composition(formula):
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a nonempty chemical formula, e.g. SiO2")
    formula = ''.join(formula.split())
    position = 0

    def count():
        nonlocal position
        match = _NUMBER.match(formula, position)
        if match is None:
            return 1.0
        position = match.end()
        value = float(match.group())
        if not np.isfinite(value) or value <= 0:
            raise ValueError("stoichiometric counts must be finite and positive")
        return value

    def group(nested=False):
        nonlocal position
        result = defaultdict(float)
        while position < len(formula):
            char = formula[position]
            if char == ')':
                if not nested or not result:
                    raise ValueError(f"unexpected or empty parentheses in {formula!r}")
                position += 1
                return result
            if char == '(':
                position += 1
                members = group(nested=True)
            else:
                match = _ELEMENT.match(formula, position)
                if match is None:
                    raise ValueError(f"invalid formula {formula!r} at position {position}")
                element = match.group()
                if element not in _ATOMIC_WEIGHTS:
                    raise ValueError(f"unknown or unsupported element {element!r}; tables cover H through U")
                position = match.end()
                members = {element: 1.0}
            multiplier = count()
            for element, amount in members.items():
                result[element] += amount * multiplier
                if not np.isfinite(result[element]):
                    raise ValueError("stoichiometric count overflow")
        if nested:
            raise ValueError(f"unclosed parentheses in {formula!r}")
        return result

    return dict(group())


def _formula_mass(composition):
    missing = [element for element in composition if _ATOMIC_WEIGHTS[element] is None]
    if missing:
        raise ValueError(f"no standard atomic weight for {', '.join(missing)}; supply molar_mass in g/mol")
    return sum(amount * _ATOMIC_WEIGHTS[element] for element, amount in composition.items())


def molar_mass(formula):
    """Formula molar mass in g/mol using CIAAW abridged atomic weights.

    Supports case-sensitive element symbols, positive integer/decimal counts,
    and nested parentheses: SiO2, Si3N4, Si0.7Ge0.3, Ca(OH)2. Isotope labels,
    charge notation, square brackets, and hydrate dot notation are unsupported.
    """
    return _formula_mass(_composition(formula))


@lru_cache(maxsize=92)
def _table(element):
    resource = files('xsw').joinpath('data', 'sf', element.lower() + '.nff')
    with resource.open('r', encoding='ascii') as stream:
        data = np.loadtxt(stream, skiprows=1)
    # Preserve supplied bytes; clean only the in-memory interpolation view.
    data = data[np.argsort(data[:, 0], kind='stable')]
    # A few updated tables repeat energies. Retain the last supplied row.
    data = data[np.r_[np.diff(data[:, 0]) != 0, True]]
    if not np.isfinite(data).all() or np.any(data[:, 2] < 0):
        raise ValueError(f"invalid CXRO table for {element}")
    data.setflags(write=False)
    return data


def _positive_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    values = np.asarray(value, dtype=float)
    if not values.size or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return values


@overload
def scattering_factors(element: str, energy: float | int) -> tuple[float, float]: ...
@overload
def scattering_factors(element: str, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
def scattering_factors(element, energy):
    """Return (f1, f2) for one element at energy in eV (scalar or array).

    Piecewise linear interpolation in energy; no extrapolation. Intervals
    touching an undefined f1=-9999 row are rejected, not bridged or clamped.
    """
    if not isinstance(element, str) or element not in _ATOMIC_WEIGHTS:
        raise ValueError("element must be a case-sensitive symbol from H through U")
    energy = _positive_array(energy, 'energy')
    data = _table(element)
    grid = data[:, 0]
    if np.any((energy < grid[0]) | (energy > grid[-1])):
        raise ValueError(f"{element} table covers {grid[0]:g} to {grid[-1]:g} eV; no extrapolation")
    hi = np.searchsorted(grid, energy, side='left')
    lo = np.where(energy == grid[hi], hi, np.maximum(hi-1, 0))
    if np.any(data[lo, 1] == -9999) or np.any(data[hi, 1] == -9999):
        raise ValueError(f"{element}: f1 is undefined at or around the requested energy (-9999 in CXRO table)")
    f1 = np.interp(energy.ravel(), grid, data[:, 1]).reshape(energy.shape)
    f2 = np.interp(energy.ravel(), grid, data[:, 2]).reshape(energy.shape)
    if energy.ndim == 0:
        return float(f1), float(f2)
    return f1, f2


@overload
def chi_from_formula(
    formula: str,
    density: float,
    *,
    energy: float | int,
    wavelength: None = None,
    molar_mass: float | None = None,
) -> complex: ...
@overload
def chi_from_formula(
    formula: str,
    density: float,
    *,
    energy: None = None,
    wavelength: float | int,
    molar_mass: float | None = None,
) -> complex: ...
@overload
def chi_from_formula(
    formula: str,
    density: float,
    *,
    energy: np.ndarray,
    wavelength: None = None,
    molar_mass: float | None = None,
) -> np.ndarray: ...
@overload
def chi_from_formula(
    formula: str,
    density: float,
    *,
    energy: None = None,
    wavelength: np.ndarray,
    molar_mass: float | None = None,
) -> np.ndarray: ...
def chi_from_formula(formula, density, *, energy=None, wavelength=None, molar_mass=None):
    """Complex dielectric susceptibility chi=epsilon_r-1 for a compound.

    density: mass density in g/cm^3 (scalar, >=0).
    energy: photon energy in eV; or wavelength: vacuum wavelength in nm.
    molar_mass: optional formula molar mass in g/mol, overriding atomic weights.

    Exactly one of energy/wavelength is required; arrays return same-shape
    complex arrays, scalars return complex. The independent-atom X-ray formula
    gives positive Im(chi) with this package's exp(-i omega t) convention.
    This density is distinct from Material.density's fluorescence weight.
    """
    composition = _composition(formula)
    density_array = np.asarray(density)
    if (density_array.ndim != 0 or np.iscomplexobj(density_array)
            or not np.isfinite(density_array) or density_array < 0):
        raise ValueError("density must be a finite nonnegative scalar in g/cm^3")
    if (energy is None) == (wavelength is None):
        raise ValueError("provide exactly one of energy (eV) or wavelength (nm)")
    if energy is not None:
        energy = _positive_array(energy, 'energy')
        wavelength = HC_EV_NM / energy
    else:
        wavelength = _positive_array(wavelength, 'wavelength')
        energy = HC_EV_NM / wavelength
    mass = _formula_mass(composition) if molar_mass is None else _positive_array(molar_mass, 'molar_mass')
    if np.ndim(mass) != 0:
        raise ValueError("molar_mass must be a scalar in g/mol")
    factor = np.zeros(energy.shape, complex)
    for element, amount in composition.items():
        f1, f2 = scattering_factors(element, energy)
        factor += amount * (f1 - 1j * f2)
    electron_radius_nm = physical_constants['classical electron radius'][0] * 1e9
    number_density_nm3 = float(density_array) * Avogadro / float(mass) * 1e-21
    chi = -electron_radius_nm * wavelength**2 / np.pi * number_density_nm3 * factor
    return complex(chi) if energy.ndim == 0 else chi
