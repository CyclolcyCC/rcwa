"""Forward simulation, analytical scale evaluation, and weighted chi^2 residuals."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

import xsw
from .data import ExperimentData
from .divergence import convolve_divergence
from .geometry import AndrleBParams, build_andrle_b_sample


# Attenuation coefficients at E = 680 eV excitation in nm^-1
# mu = 2*pi * beta / lambda = 2*pi * (chi.imag / 2) / lambda (or mac * rho * 1e-7)
DEFAULT_ATTENUATION_N = 2.75e-3   # N-Kalpha (392.4 eV)
DEFAULT_ATTENUATION_O = 2.33e-3   # O-Kalpha (524.9 eV)
DEFAULT_ATTENUATION_C = 6.47e-4   # C-Kalpha (277.0 eV)
WAVELENGTH_680EV = 1.8233         # nm


@dataclass
class ForwardResult:
    """Detailed forward calculation outputs and fit metrics."""

    angles_deg: np.ndarray
    raw_N: np.ndarray
    raw_O: np.ndarray
    conv_N: np.ndarray
    conv_O: np.ndarray
    scaled_N: np.ndarray
    scaled_O: np.ndarray
    scale_N: float
    scale_O: float
    res_N: np.ndarray
    res_O: np.ndarray
    chi2_N: float
    chi2_O: float
    total_chi2: float
    dof: int
    reduced_chi2: float

    # Carbon channel (optional)
    raw_C: Optional[np.ndarray] = None
    conv_C: Optional[np.ndarray] = None
    scaled_C: Optional[np.ndarray] = None
    scale_C: Optional[float] = None
    res_C: Optional[np.ndarray] = None
    chi2_C: Optional[float] = None


def compute_optimal_scale(exp: np.ndarray, model: np.ndarray, sigma: np.ndarray) -> float:
    """Compute analytical weighted least-squares scaling factor c:

    min_c sum [ (exp - c * model) / sigma ]^2
    => c = sum(exp * model / sigma^2) / sum(model^2 / sigma^2)
    """
    denom = np.sum((model / sigma) ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return 1.0
    num = np.sum((exp * model) / (sigma ** 2))
    scale = float(num / denom)
    return max(0.0, scale)


def evaluate_forward(
    params: AndrleBParams | np.ndarray,
    data: ExperimentData,
    *,
    num_slices: int = 10,
    max_order: int = 12,
    polarization: str = "p",
    divergence_sigma_deg: float = 0.01,
    attenuation_N: float = DEFAULT_ATTENUATION_N,
    attenuation_O: float = DEFAULT_ATTENUATION_O,
    attenuation_C: float = DEFAULT_ATTENUATION_C,
) -> ForwardResult:
    """Run full forward RCWA simulation and evaluate weighted residuals and chi^2.

    Parameters
    ----------
    params : AndrleBParams or array-like
        Candidate structure parameters.
    data : ExperimentData
        Experimental dataset containing measured angles, yields, and uncertainties.
    num_slices : int
        Multislice slice count for the trapezoid line.
    max_order : int
        Maximum Fourier order (-M to M, total 2M+1 harmonics).
    polarization : str
        Input polarization state ('p' or 's').
    divergence_sigma_deg : float
        X-ray beam divergence standard deviation in degrees.
    attenuation_N, attenuation_O, attenuation_C : float
        Linear escape attenuation coefficients in nm^-1.

    Returns
    -------
    ForwardResult
        Dataclass containing raw yields, convolved yields, scale factors, and chi^2.
    """
    structure, mat_n, mat_o, mat_c = build_andrle_b_sample(params, num_slices=num_slices)
    prepared = xsw.prepare(structure, max_order=max_order)

    raw_n = np.empty(data.num_angles, dtype=float)
    raw_o = np.empty(data.num_angles, dtype=float)
    raw_c = np.empty(data.num_angles, dtype=float)

    for i, alpha_rad in enumerate(data.angles_rad):
        sol = prepared.solve(alpha=alpha_rad, phi=0.0, wavelength=WAVELENGTH_680EV,
                             polarization=polarization)
        raw_n[i] = sol.fluorescence(mat_n, attenuation=attenuation_N)
        raw_o[i] = sol.fluorescence(mat_o, attenuation=attenuation_O)
        raw_c[i] = sol.fluorescence(mat_c, attenuation=attenuation_C)

    # Beam divergence convolution
    conv_n = convolve_divergence(data.angles_deg, raw_n, divergence_sigma_deg)
    conv_o = convolve_divergence(data.angles_deg, raw_o, divergence_sigma_deg)
    conv_c = convolve_divergence(data.angles_deg, raw_c, divergence_sigma_deg)

    # Analytical optimal scaling for N and O
    c_n = compute_optimal_scale(data.exp_N, conv_n, data.sigma_N)
    c_o = compute_optimal_scale(data.exp_O, conv_o, data.sigma_O)

    scaled_n = c_n * conv_n
    scaled_o = c_o * conv_o

    res_n = (data.exp_N - scaled_n) / data.sigma_N
    res_o = (data.exp_O - scaled_o) / data.sigma_O

    chi2_n = float(np.sum(res_n ** 2))
    chi2_o = float(np.sum(res_o ** 2))

    # Analytical optimal scaling for C (if present in dataset)
    if data.has_carbon and data.exp_C is not None and data.sigma_C is not None:
        c_c = compute_optimal_scale(data.exp_C, conv_c, data.sigma_C)
        scaled_c = c_c * conv_c
        res_c = (data.exp_C - scaled_c) / data.sigma_C
        chi2_c = float(np.sum(res_c ** 2))
        total_chi2 = chi2_n + chi2_o + chi2_c
        n_channels = 3
    else:
        c_c = None
        scaled_c = None
        res_c = None
        chi2_c = None
        total_chi2 = chi2_n + chi2_o
        n_channels = 2

    num_params = len(params.as_array()) if isinstance(params, AndrleBParams) else len(params)
    dof = max(1, n_channels * data.num_angles - num_params)
    reduced_chi2 = total_chi2 / dof

    return ForwardResult(
        angles_deg=data.angles_deg,
        raw_N=raw_n,
        raw_O=raw_o,
        conv_N=conv_n,
        conv_O=conv_o,
        scaled_N=scaled_n,
        scaled_O=scaled_o,
        scale_N=c_n,
        scale_O=c_o,
        res_N=res_n,
        res_O=res_o,
        chi2_N=chi2_n,
        chi2_O=chi2_o,
        total_chi2=total_chi2,
        dof=dof,
        reduced_chi2=reduced_chi2,
        raw_C=raw_c,
        conv_C=conv_c,
        scaled_C=scaled_c,
        scale_C=c_c,
        res_C=res_c,
        chi2_C=chi2_c,
    )


def compute_residuals(
    param_vector: np.ndarray,
    data: ExperimentData,
    **kwargs,
) -> np.ndarray:
    """Residual function for scipy.optimize.least_squares.

    Returns concatenated 1D vector [res_N, res_O] or [res_N, res_O, res_C]
    such that sum(residuals**2) = chi^2.
    """
    try:
        result = evaluate_forward(param_vector, data, **kwargs)
        if data.has_carbon and result.res_C is not None:
            return np.concatenate([result.res_N, result.res_O, result.res_C])
        return np.concatenate([result.res_N, result.res_O])
    except Exception:
        # Return large penalizing residuals if geometry is invalid
        n_channels = 3 if data.has_carbon else 2
        return np.full(n_channels * data.num_angles, 1e4)
