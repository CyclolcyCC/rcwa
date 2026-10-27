"""Optimization engine for reconstructing Configuration B parameters via chi^2 minimization."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.optimize import least_squares

from .data import ExperimentData
from .geometry import AndrleBParams
from .objective import ForwardResult, compute_residuals, evaluate_forward


# Physical search bounds for [h, w, swa, tt, tg, sigma_sw, sigma_vert, tc]
DEFAULT_LOWER_BOUNDS_8 = np.array([80.0, 35.0, 78.0, 1.0, 2.0, 0.0, 0.0, 0.5], dtype=float)
DEFAULT_UPPER_BOUNDS_8 = np.array([110.0, 55.0, 88.0, 4.5, 8.0, 3.5, 2.5, 5.0], dtype=float)

DEFAULT_LOWER_BOUNDS_7 = DEFAULT_LOWER_BOUNDS_8[:7]
DEFAULT_UPPER_BOUNDS_7 = DEFAULT_UPPER_BOUNDS_8[:7]

DEFAULT_LOWER_BOUNDS_5 = DEFAULT_LOWER_BOUNDS_8[:5]
DEFAULT_UPPER_BOUNDS_5 = DEFAULT_UPPER_BOUNDS_8[:5]

DEFAULT_LOWER_BOUNDS = DEFAULT_LOWER_BOUNDS_7
DEFAULT_UPPER_BOUNDS = DEFAULT_UPPER_BOUNDS_7


@dataclass
class FitResult:
    """Summary of the optimization results and reconstructed structure."""

    params: AndrleBParams
    param_errors: np.ndarray
    covariance: np.ndarray
    forward: ForwardResult
    chi2: float
    reduced_chi2: float
    success: bool
    message: str
    nfev: int

    def summary(self) -> str:
        """Formatted parameter table with 1-sigma confidence intervals."""
        all_names = ["h (nm)", "w (nm)", "swa (deg)", "tt (nm)", "tg (nm)",
                     "sigma_sw (nm)", "sigma_vert (nm)", "tc (nm)"]
        vals = self.params.as_array(include_tc=True)
        errs = self.param_errors
        num_fitted = len(errs)

        scales_str = f"c_N = {self.forward.scale_N:.4e}, c_O = {self.forward.scale_O:.4e}"
        if self.forward.scale_C is not None:
            scales_str += f", c_C = {self.forward.scale_C:.4e}"

        lines = [
            "=== Reconstruction Summary (Anna Andrle Configuration B, 680 eV) ===",
            f"Optimizer Status: {self.message} (success={self.success}, evaluations={self.nfev})",
            f"Final Chi^2: {self.chi2:.2f} (Reduced Chi^2 / DoF: {self.reduced_chi2:.2f})",
            f"Optimal Scale Factors: {scales_str}",
            "-" * 65,
            f"{'Parameter':<20} | {'Value':<15} | {'1-sigma Uncertainty':<15}",
            "-" * 65,
        ]
        for i, (name, val) in enumerate(zip(all_names, vals)):
            if i < num_fitted:
                err = errs[i]
                err_str = f"+/- {err:.3f}" if np.isfinite(err) else "N/A"
            else:
                err_str = "(Fixed)"
            lines.append(f"{name:<20} | {val:<15.3f} | {err_str:<15}")
        lines.append("-" * 65)
        return "\n".join(lines)

    def save(self, filepath: str | Path, save_summary_text: bool = True) -> Path:
        """Save fitting results to a compressed .npz archive and companion text summary.

        Parameters
        ----------
        filepath : str or Path
            Target file path (automatically appends .npz if missing).
        save_summary_text : bool
            Whether to write a companion .txt file with the formatted summary table.

        Returns
        -------
        Path
            Path to the saved .npz archive.
        """
        path = Path(filepath)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        data_dict = {
            # Structural parameters
            "h": self.params.h,
            "w": self.params.w,
            "swa": self.params.swa,
            "tt": self.params.tt,
            "tg": self.params.tg,
            "sigma_sw": self.params.sigma_sw,
            "sigma_vert": self.params.sigma_vert,
            "tc": self.params.tc,
            "tsub": self.params.tsub,
            "pitch": self.params.pitch,

            # Uncertainties & Covariance
            "param_errors": np.asarray(self.param_errors),
            "covariance": np.asarray(self.covariance),

            # Metrics
            "chi2": float(self.chi2),
            "reduced_chi2": float(self.reduced_chi2),
            "success": bool(self.success),
            "message": str(self.message),
            "nfev": int(self.nfev),

            # Forward observables
            "angles_deg": self.forward.angles_deg,
            "raw_N": self.forward.raw_N,
            "raw_O": self.forward.raw_O,
            "conv_N": self.forward.conv_N,
            "conv_O": self.forward.conv_O,
            "scaled_N": self.forward.scaled_N,
            "scaled_O": self.forward.scaled_O,
            "scale_N": float(self.forward.scale_N),
            "scale_O": float(self.forward.scale_O),
            "res_N": self.forward.res_N,
            "res_O": self.forward.res_O,
            "chi2_N": float(self.forward.chi2_N),
            "chi2_O": float(self.forward.chi2_O),
            "total_chi2": float(self.forward.total_chi2),
            "dof": int(self.forward.dof),
            "forward_reduced_chi2": float(self.forward.reduced_chi2),
        }

        # Carbon channel (if present)
        if self.forward.raw_C is not None:
            data_dict["raw_C"] = self.forward.raw_C
            data_dict["conv_C"] = self.forward.conv_C
            data_dict["scaled_C"] = self.forward.scaled_C
            data_dict["res_C"] = self.forward.res_C
            data_dict["scale_C"] = float(self.forward.scale_C)
            data_dict["chi2_C"] = float(self.forward.chi2_C)

        np.savez_compressed(path, **data_dict)

        if save_summary_text:
            txt_path = path.with_suffix(".txt")
            txt_path.write_text(self.summary(), encoding="utf-8")

        return path

    @classmethod
    def load(cls, filepath: str | Path) -> "FitResult":
        """Load a FitResult from a .npz archive saved by save().

        Parameters
        ----------
        filepath : str or Path
            Path to the .npz archive.

        Returns
        -------
        FitResult
            Reconstituted FitResult object.
        """
        path = Path(filepath)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        if not path.exists():
            raise FileNotFoundError(f"Fit result file not found: {path}")

        archive = np.load(path)

        params = AndrleBParams(
            h=float(archive["h"]),
            w=float(archive["w"]),
            swa=float(archive["swa"]),
            tt=float(archive["tt"]),
            tg=float(archive["tg"]),
            sigma_sw=float(archive["sigma_sw"]),
            sigma_vert=float(archive["sigma_vert"]),
            tc=float(archive["tc"]) if "tc" in archive else 2.5,
            tsub=float(archive["tsub"]) if "tsub" in archive else 0.0,
            pitch=float(archive["pitch"]) if "pitch" in archive else 100.834,
        )

        has_c = "scaled_C" in archive
        fwd = ForwardResult(
            angles_deg=archive["angles_deg"],
            raw_N=archive["raw_N"],
            raw_O=archive["raw_O"],
            conv_N=archive["conv_N"],
            conv_O=archive["conv_O"],
            scaled_N=archive["scaled_N"],
            scaled_O=archive["scaled_O"],
            scale_N=float(archive["scale_N"]),
            scale_O=float(archive["scale_O"]),
            res_N=archive["res_N"],
            res_O=archive["res_O"],
            chi2_N=float(archive["chi2_N"]),
            chi2_O=float(archive["chi2_O"]),
            total_chi2=float(archive["total_chi2"]),
            dof=int(archive["dof"]),
            reduced_chi2=float(archive["forward_reduced_chi2"] if "forward_reduced_chi2" in archive else archive["reduced_chi2"]),
            raw_C=archive["raw_C"] if has_c else None,
            conv_C=archive["conv_C"] if has_c else None,
            scaled_C=archive["scaled_C"] if has_c else None,
            scale_C=float(archive["scale_C"]) if has_c else None,
            res_C=archive["res_C"] if has_c else None,
            chi2_C=float(archive["chi2_C"]) if has_c else None,
        )

        return cls(
            params=params,
            param_errors=archive["param_errors"],
            covariance=archive["covariance"],
            forward=fwd,
            chi2=float(archive["chi2"]),
            reduced_chi2=float(archive["reduced_chi2"]),
            success=bool(archive["success"]),
            message=str(archive["message"]),
            nfev=int(archive["nfev"]),
        )


def fit_andrle_structure(
    data: ExperimentData,
    initial_params: Optional[AndrleBParams | np.ndarray] = None,
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
    *,
    num_slices: int = 10,
    max_order: int = 12,
    divergence_sigma_deg: float = 0.01,
    max_nfev: int = 100,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    diff_step: float = 1e-3,
    verbose: int = 2,
) -> FitResult:
    """Fit Configuration B parameters by minimizing weighted chi^2 using least_squares.

    Parameters
    ----------
    data : ExperimentData
        Measured angles, signals, and uncertainties.
    initial_params : AndrleBParams or array-like, optional
        Starting guess. Defaults to nominal literature values:
        h=93.0, w=44.9, swa=82.5, tt=2.47, tg=3.67, sigma_sw=1.5, sigma_vert=1.0, tc=2.5.
    bounds : tuple of (lower_bounds, upper_bounds), optional
        Box bounds on parameters (supports 5, 7, or 8 parameters).
    num_slices : int
        Number of multislice layers for grating trapezoid.
    max_order : int
        Max RCWA spatial harmonic order.
    divergence_sigma_deg : float
        Beamline angular divergence (nominal <= 0.01 deg).
    max_nfev : int
        Maximum number of forward model evaluations.
    ftol, xtol : float
        Tolerance thresholds for termination.
    diff_step : float
        Relative step size for finite-difference Jacobian approximation.
    verbose : int
        Verbosity level (0: silent, 1: end summary, 2: print each iteration).

    Returns
    -------
    FitResult
        Container with fitted parameters, uncertainties, final chi^2, and curves.
    """
    if initial_params is None:
        p0 = AndrleBParams()
    elif isinstance(initial_params, AndrleBParams):
        p0 = initial_params
    else:
        init_arr = np.asarray(initial_params, dtype=float)
        p0 = AndrleBParams.from_array(init_arr)

    # Determine parameter dimensionality
    if bounds is not None:
        lb, ub = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
        dim = len(lb)
    elif initial_params is not None and not isinstance(initial_params, AndrleBParams):
        dim = len(init_arr)
        if dim == 8:
            lb, ub = DEFAULT_LOWER_BOUNDS_8, DEFAULT_UPPER_BOUNDS_8
        elif dim == 7:
            lb, ub = DEFAULT_LOWER_BOUNDS_7, DEFAULT_UPPER_BOUNDS_7
        else:
            lb, ub = DEFAULT_LOWER_BOUNDS_5, DEFAULT_UPPER_BOUNDS_5
    else:
        # Default to 7 parameters (geometry + roughness, tc fixed at nominal 2.5 nm)
        dim = 7
        lb, ub = DEFAULT_LOWER_BOUNDS_7, DEFAULT_UPPER_BOUNDS_7

    if dim == 8:
        x0 = p0.as_array(include_tc=True)
    elif dim == 7:
        x0 = p0.as_array(include_tc=False)
    elif dim == 5:
        x0 = p0.as_array(include_tc=False)[:5]
    else:
        raise ValueError(f"Expected 5, 7, or 8 parameters, got dimensionality {dim}")

    x0 = np.clip(x0, lb + 1e-4, ub - 1e-4)

    kwargs = {
        "num_slices": num_slices,
        "max_order": max_order,
        "polarization": "p",
        "divergence_sigma_deg": divergence_sigma_deg,
    }

    def _obj_wrapper(x):
        if len(x) == 8:
            p_vec = x
        elif len(x) == 7:
            p_vec = np.array([x[0], x[1], x[2], x[3], x[4], x[5], x[6], p0.tc], dtype=float)
        elif len(x) == 5:
            p_vec = np.array([x[0], x[1], x[2], x[3], x[4], p0.sigma_sw, p0.sigma_vert, p0.tc], dtype=float)
        else:
            p_vec = x
        return compute_residuals(p_vec, data, **kwargs)

    opt = least_squares(
        _obj_wrapper,
        x0=x0,
        bounds=(lb, ub),
        method="trf",
        diff_step=diff_step,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        verbose=verbose,
    )

    if dim == 8:
        fitted_p = AndrleBParams.from_array(opt.x, tsub=p0.tsub, pitch=p0.pitch)
    elif dim == 7:
        fitted_p = AndrleBParams(
            h=opt.x[0], w=opt.x[1], swa=opt.x[2], tt=opt.x[3], tg=opt.x[4],
            sigma_sw=opt.x[5], sigma_vert=opt.x[6], tc=p0.tc,
            tsub=p0.tsub, pitch=p0.pitch,
        )
    else:
        fitted_p = AndrleBParams(
            h=opt.x[0], w=opt.x[1], swa=opt.x[2], tt=opt.x[3], tg=opt.x[4],
            sigma_sw=p0.sigma_sw, sigma_vert=p0.sigma_vert, tc=p0.tc,
            tsub=p0.tsub, pitch=p0.pitch,
        )

    final_forward = evaluate_forward(
        fitted_p,
        data,
        num_slices=num_slices,
        max_order=max_order,
        polarization="p",
        divergence_sigma_deg=divergence_sigma_deg,
    )

    # Calculate covariance matrix from Jacobian: Cov = inv(J^T J) * (chi^2 / DoF)
    j = opt.jac
    try:
        jt_j = j.T @ j
        cov = np.linalg.pinv(jt_j) * final_forward.reduced_chi2
        param_errors = np.sqrt(np.maximum(0.0, np.diag(cov)))
    except Exception:
        cov = np.full((len(x0), len(x0)), np.nan)
        param_errors = np.full(len(x0), np.nan)

    return FitResult(
        params=fitted_p,
        param_errors=param_errors,
        covariance=cov,
        forward=final_forward,
        chi2=final_forward.total_chi2,
        reduced_chi2=final_forward.reduced_chi2,
        success=bool(opt.success),
        message=str(opt.message),
        nfev=int(opt.nfev),
    )
