"""Container and loader for Anna Andrle Configuration B (680 eV) experimental GIXRF data."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


def _find_default_data_file() -> Path:
    """Locate 20200417_BestFit.npy in standard locations."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "legacy" / "self_attenuation" / "20200417_BestFit.npy"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate 20200417_BestFit.npy in repository.")


@dataclass(frozen=True)
class ExperimentData:
    """Experimental GIXRF data container for N, O, and optional C fluorescence.

    All arrays have length N (measured angles).
    """

    angles_deg: np.ndarray
    exp_N: np.ndarray
    exp_O: np.ndarray
    sigma_N: np.ndarray
    sigma_O: np.ndarray
    exp_C: Optional[np.ndarray] = None
    sigma_C: Optional[np.ndarray] = None
    jcm_N: Optional[np.ndarray] = None
    jcm_O: Optional[np.ndarray] = None
    jcm_C: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.angles_deg)
        required = [('exp_N', self.exp_N), ('exp_O', self.exp_O),
                    ('sigma_N', self.sigma_N), ('sigma_O', self.sigma_O)]
        if self.exp_C is not None:
            required.append(('exp_C', self.exp_C))
        if self.sigma_C is not None:
            required.append(('sigma_C', self.sigma_C))

        for name, arr in required:
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} does not match angles length {n}")
            if np.any(arr <= 0) or not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain finite positive values")

    @property
    def has_carbon(self) -> bool:
        """Whether carbon channel data is present."""
        return self.exp_C is not None and self.sigma_C is not None

    @property
    def angles_rad(self) -> np.ndarray:
        """Grazing incidence angles in radians."""
        return np.deg2rad(self.angles_deg)

    @property
    def num_angles(self) -> int:
        """Total number of measured angular points."""
        return len(self.angles_deg)

    def subsample(self, step: int = 1) -> "ExperimentData":
        """Return a subsampled dataset taking every step-th angle point."""
        if step < 1:
            raise ValueError("step must be a positive integer")
        sl = slice(None, None, step)
        return ExperimentData(
            angles_deg=self.angles_deg[sl],
            exp_N=self.exp_N[sl],
            exp_O=self.exp_O[sl],
            sigma_N=self.sigma_N[sl],
            sigma_O=self.sigma_O[sl],
            exp_C=self.exp_C[sl] if self.exp_C is not None else None,
            sigma_C=self.sigma_C[sl] if self.sigma_C is not None else None,
            jcm_N=self.jcm_N[sl] if self.jcm_N is not None else None,
            jcm_O=self.jcm_O[sl] if self.jcm_O is not None else None,
            jcm_C=self.jcm_C[sl] if self.jcm_C is not None else None,
        )


def load_andrle_data(filepath: Optional[str | Path] = None) -> ExperimentData:
    """Load experimental data for Configuration B from a .npy or .npz file.

    Supports:
    - Standard 7-row legacy .npy format (20200417_BestFit.npy)
    - 9- or 10-row .npy format including carbon
    - .npz archive with dictionary keys (angles, N_exp, N_err, O_exp, O_err, C_exp, C_err)

    Parameters
    ----------
    filepath : str or Path, optional
        Path to file. If omitted, locates 20200417_BestFit.npy in legacy/self_attenuation/.

    Returns
    -------
    ExperimentData
        Container with angles (deg), experimental counts, uncertainties, and references.
    """
    path = Path(filepath) if filepath is not None else _find_default_data_file()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        keys = list(archive.keys())

        # Extract angles
        angles = archive["angles"] if "angles" in keys else archive["angles_deg"]

        # Extract N
        exp_n = archive["N_exp"] if "N_exp" in keys else archive["exp_N"]
        sig_n = archive["N_err"] if "N_err" in keys else (archive["sigma_N"] if "sigma_N" in keys else archive["N_sigma"])

        # Extract O
        exp_o = archive["O_exp"] if "O_exp" in keys else archive["exp_O"]
        sig_o = archive["O_err"] if "O_err" in keys else (archive["sigma_O"] if "sigma_O" in keys else archive["O_sigma"])

        # Extract C if available
        exp_c = None
        sig_c = None
        if "C_exp" in keys:
            exp_c = archive["C_exp"].copy()
        elif "exp_C" in keys:
            exp_c = archive["exp_C"].copy()

        if "C_err" in keys:
            sig_c = archive["C_err"].copy()
        elif "sigma_C" in keys:
            sig_c = archive["sigma_C"].copy()
        elif "C_sigma" in keys:
            sig_c = archive["C_sigma"].copy()

        jcm_n = archive["jcm_N"].copy() if "jcm_N" in keys else None
        jcm_o = archive["jcm_O"].copy() if "jcm_O" in keys else None
        jcm_c = archive["jcm_C"].copy() if "jcm_C" in keys else None

        return ExperimentData(
            angles_deg=angles.copy(),
            exp_N=exp_n.copy(),
            exp_O=exp_o.copy(),
            sigma_N=sig_n.copy(),
            sigma_O=sig_o.copy(),
            exp_C=exp_c,
            sigma_C=sig_c,
            jcm_N=jcm_n,
            jcm_O=jcm_o,
            jcm_C=jcm_c,
        )

    raw = np.load(path)
    if raw.ndim == 2 and raw.shape[0] >= 9:
        # Format with carbon: [angles, exp_N, exp_O, exp_C, jcm_N, jcm_O, jcm_C, sigma_N, sigma_O, sigma_C]
        angles = raw[0].copy()
        exp_n = raw[1].copy()
        exp_o = raw[2].copy()
        exp_c = raw[3].copy()
        if raw.shape[0] >= 10:
            jcm_n = raw[4].copy()
            jcm_o = raw[5].copy()
            jcm_c = raw[6].copy()
            sig_n = raw[7].copy()
            sig_o = raw[8].copy()
            sig_c = raw[9].copy()
        else:
            jcm_n, jcm_o, jcm_c = None, None, None
            sig_n = raw[4].copy()
            sig_o = raw[5].copy()
            sig_c = raw[6].copy()

        return ExperimentData(
            angles_deg=angles,
            exp_N=exp_n,
            exp_O=exp_o,
            sigma_N=sig_n,
            sigma_O=sig_o,
            exp_C=exp_c,
            sigma_C=sig_c,
            jcm_N=jcm_n,
            jcm_O=jcm_o,
            jcm_C=jcm_c,
        )

    if raw.shape[0] < 7:
        raise ValueError(f"Expected array with at least 7 rows, got shape {raw.shape}")

    return ExperimentData(
        angles_deg=raw[0].copy(),
        exp_N=raw[1].copy(),
        exp_O=raw[2].copy(),
        jcm_N=raw[3].copy(),
        jcm_O=raw[4].copy(),
        sigma_N=raw[5].copy(),
        sigma_O=raw[6].copy(),
    )
