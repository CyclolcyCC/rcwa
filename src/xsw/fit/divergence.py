"""Gaussian convolution for beamline angular divergence."""
import numpy as np


def divergence_matrix(angles_deg: np.ndarray, sigma_deg: float = 0.01) -> np.ndarray:
    """Construct normalized convolution matrix for Gaussian angular divergence.

    Parameters
    ----------
    angles_deg : np.ndarray
        Array of incident grazing angles in degrees.
    sigma_deg : float, optional
        Standard deviation of the beam angular divergence in degrees.
        If sigma_deg <= 0, returns the identity matrix (no divergence).

    Returns
    -------
    np.ndarray
        Square matrix M of shape (N, N) such that y_conv = M @ y.
    """
    angles = np.asarray(angles_deg, dtype=float)
    if angles.ndim != 1 or len(angles) == 0:
        raise ValueError("angles_deg must be a non-empty 1D array")
    if sigma_deg <= 0.0:
        return np.eye(len(angles))

    # Pairwise angular difference matrix: d_alpha[i, j] = alpha[i] - alpha[j]
    d_alpha = angles[:, None] - angles[None, :]
    kernel = np.exp(-0.5 * (d_alpha / sigma_deg) ** 2)

    # Trapezoidal quadrature weights along alpha_j
    if len(angles) > 1:
        # Approximate cell widths around each point
        diffs = np.diff(angles)
        weights = np.empty_like(angles)
        weights[0] = 0.5 * diffs[0]
        weights[-1] = 0.5 * diffs[-1]
        weights[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    else:
        weights = np.ones(1)

    weighted_kernel = kernel * weights[None, :]
    row_sums = np.sum(weighted_kernel, axis=1, keepdims=True)
    # Normalize each row to sum to 1 so constant signal and total flux are conserved
    return weighted_kernel / row_sums


def convolve_divergence(angles_deg: np.ndarray, yields: np.ndarray,
                        sigma_deg: float = 0.01) -> np.ndarray:
    """Convolve 1D angular fluorescence yields with Gaussian beam divergence.

    Parameters
    ----------
    angles_deg : np.ndarray
        Incident grazing angles in degrees.
    yields : np.ndarray
        Calculated fluorescence yields, shape (N,).
    sigma_deg : float
        Beam angular spread standard deviation in degrees (nominal <= 0.01 deg).

    Returns
    -------
    np.ndarray
        Convolved angular yield curve of same shape (N,).
    """
    if sigma_deg <= 0.0:
        return np.asarray(yields, dtype=float).copy()
    mat = divergence_matrix(angles_deg, sigma_deg)
    return mat @ np.asarray(yields, dtype=float)
