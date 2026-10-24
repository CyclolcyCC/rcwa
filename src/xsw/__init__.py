"""Vector X-ray standing waves in gratings invariant along y."""
from .model import Box, Layer, Material, Structure, VACUUM
from .solver import HC_EV_NM, PreparedStructure, Solution, prepare, solve
from .observables import normalize
from .scan import fluorescence_map
from .optical_constants import chi_from_formula, molar_mass, scattering_factors

__version__ = "0.1.0"
__all__ = ["Box", "Layer", "Material", "Structure", "VACUUM", "HC_EV_NM",
           "PreparedStructure", "Solution", "prepare", "solve", "normalize", "fluorescence_map",
           "chi_from_formula", "molar_mass", "scattering_factors"]
