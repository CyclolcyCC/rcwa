"""Parametric trapezoidal geometry builder for Anna Andrle Configuration B."""
from dataclasses import dataclass
import numpy as np
import xsw


# Default calibrated optical constants at E = 680 eV (lambda = 1.8233 nm)
# Sourced from PTB calibration (Nanomaterials 2021, 11, 1647 and thesis Section 4.3 & 4.5)
DEFAULT_CHI_SI = complex(-1.984e-3, 3.046e-4)
DEFAULT_CHI_SI3N4 = complex(-2.518e-3, 5.166e-4)
DEFAULT_CHI_SIO2 = complex(-1.746e-3, 5.163e-4)
DEFAULT_CHI_C2O = complex(-1.590e-3, 4.537e-4)  # C2O at rho = 1.8 g/cm^3 from CXRO


@dataclass
class AndrleBParams:
    """Nanoscale geometric and roughness parameters for Configuration B."""

    h: float = 93.0            # Total height of the nitride line (nm)
    w: float = 44.9            # Middle line width (CD at half height) (nm)
    swa: float = 82.5          # Sidewall angle (degrees)
    tt: float = 2.47           # Top oxide cap thickness (nm)
    tg: float = 3.67           # Groove oxide thickness (nm)
    sigma_sw: float = 1.5      # Debye-Waller sidewall roughness (nm)
    sigma_vert: float = 1.0    # Nevot-Croce interface roughness (nm)
    tc: float = 2.5            # Surface contamination (C2O) layer thickness (nm, nominal 2.5)
    tsub: float = 0.0          # Substrate native oxide thickness (nm, default 0.0: excluded)
    pitch: float = 100.834     # Grating pitch (nm)

    def as_array(self, include_tc: bool = True) -> np.ndarray:
        """Return parameters as a 1D vector for optimizers:

        If include_tc is True: [h, w, swa, tt, tg, sigma_sw, sigma_vert, tc] (8 elements).
        If include_tc is False: [h, w, swa, tt, tg, sigma_sw, sigma_vert] (7 elements).
        """
        if include_tc:
            return np.array([self.h, self.w, self.swa, self.tt, self.tg,
                             self.sigma_sw, self.sigma_vert, self.tc], dtype=float)
        return np.array([self.h, self.w, self.swa, self.tt, self.tg,
                         self.sigma_sw, self.sigma_vert], dtype=float)

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        tc: float = 2.5,
        tsub: float = 0.0,
        pitch: float = 100.834,
    ) -> "AndrleBParams":
        """Construct AndrleBParams from an array of 5, 6, 7, or 8 values."""
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 5:
            # [h, w, swa, tt, tg]
            return cls(h=arr[0], w=arr[1], swa=arr[2], tt=arr[3], tg=arr[4],
                       sigma_sw=0.0, sigma_vert=0.0, tc=tc, tsub=tsub, pitch=pitch)
        elif len(arr) == 6:
            # [h, w, swa, tt, tg, tc]
            return cls(h=arr[0], w=arr[1], swa=arr[2], tt=arr[3], tg=arr[4],
                       sigma_sw=0.0, sigma_vert=0.0, tc=arr[5], tsub=tsub, pitch=pitch)
        elif len(arr) == 7:
            # [h, w, swa, tt, tg, sigma_sw, sigma_vert]
            return cls(h=arr[0], w=arr[1], swa=arr[2], tt=arr[3], tg=arr[4],
                       sigma_sw=arr[5], sigma_vert=arr[6], tc=tc, tsub=tsub, pitch=pitch)
        elif len(arr) == 8:
            # [h, w, swa, tt, tg, sigma_sw, sigma_vert, tc]
            return cls(h=arr[0], w=arr[1], swa=arr[2], tt=arr[3], tg=arr[4],
                       sigma_sw=arr[5], sigma_vert=arr[6], tc=arr[7], tsub=tsub, pitch=pitch)
        else:
            raise ValueError(f"Expected array of 5, 6, 7, or 8 parameters, got length {len(arr)}")


def build_andrle_b_sample(
    params: AndrleBParams | np.ndarray,
    num_slices: int = 10,
    optical_constants: tuple[complex, complex, complex] | tuple[complex, complex, complex, complex] | None = None,
) -> tuple[xsw.Structure, xsw.Material, xsw.Material, xsw.Material]:
    """Build an xsw.Structure representing Configuration B at E = 680 eV with carbon contamination.

    Parameters
    ----------
    params : AndrleBParams or array-like
        Geometric parameters: [h, w, swa, tt, tg, sigma_sw, sigma_vert, tc].
    num_slices : int, optional
        Total number of horizontal slices used to discretize the trapezoid line.
    optical_constants : tuple of susceptibilities, optional
        Custom optical susceptibilities: (chi_Si, chi_Si3N4, chi_SiO2[, chi_C2O]).
        If omitted, uses PTB calibrated & CXRO values at 680 eV.

    Returns
    -------
    structure : xsw.Structure
        Ready-to-solve multislice grating structure.
    mat_si3n4 : xsw.Material
        The fluorescent Si3N4 material object (for N-Kalpha yield integration).
    mat_sio2 : xsw.Material
        The fluorescent SiO2 material object (for O-Kalpha yield integration).
    mat_c2o : xsw.Material
        The fluorescent C2O material object (for C-Kalpha yield integration).
    """
    if not isinstance(params, AndrleBParams):
        p = AndrleBParams.from_array(params)
    else:
        p = params

    if p.h <= 0 or p.w <= 0 or p.tt < 0 or p.tg < 0 or p.tc < 0 or p.pitch <= 0:
        raise ValueError("Dimensions must be positive (thicknesses non-negative).")
    if p.tg + p.tc >= p.h:
        raise ValueError(f"Groove stack tg + tc ({p.tg + p.tc:.2f} nm) cannot exceed line height h ({p.h:.2f} nm).")
    if p.swa <= 60.0 or p.swa >= 90.0:
        raise ValueError(f"Sidewall angle {p.swa} deg must be between 60 and 90 deg.")

    # Materials
    if optical_constants is None:
        chi_si = DEFAULT_CHI_SI
        chi_si3n4 = DEFAULT_CHI_SI3N4
        chi_sio2 = DEFAULT_CHI_SIO2
        chi_c2o = DEFAULT_CHI_C2O
    elif len(optical_constants) == 3:
        chi_si, chi_si3n4, chi_sio2 = optical_constants
        chi_c2o = DEFAULT_CHI_C2O
    elif len(optical_constants) == 4:
        chi_si, chi_si3n4, chi_sio2, chi_c2o = optical_constants
    else:
        raise ValueError("optical_constants must be a tuple of 3 or 4 complex susceptibilities.")

    mat_si = xsw.Material(chi_si, name="Si")
    mat_si3n4 = xsw.Material(chi_si3n4, name="Si3N4", fluorescent=True)
    mat_sio2 = xsw.Material(chi_sio2, name="SiO2", fluorescent=True)
    mat_c2o = xsw.Material(chi_c2o, name="C2O", fluorescent=True)

    structure = xsw.Structure(period=p.pitch, substrate=mat_si, substrate_sigma=p.sigma_vert)

    # 1. Top cap widths
    top_nitride_w = p.w - (p.h / 2.0) * (2.0 / np.tan(np.deg2rad(p.swa)))
    if top_nitride_w <= 0:
        raise ValueError(f"Trapezoid top width {top_nitride_w:.2f} nm is non-positive. Check h, w, swa.")

    w_ox_cap = min(top_nitride_w + 2.0 * p.tt, p.pitch - 1e-4)
    w_c_cap = min(w_ox_cap + 2.0 * p.tc, p.pitch - 1e-4)

    # Top carbon cap (if tc > 0)
    if p.tc > 0:
        l_c_top = xsw.Layer(thickness=p.tc, sigma=p.sigma_vert)
        l_c_top.add_box(-w_c_cap / 2.0, w_c_cap / 2.0, mat_c2o)
        structure.add_layer(l_c_top)

    # Top oxide cap (if tt > 0)
    if p.tt > 0:
        sig_ox = 0.0 if p.tc > 0 else p.sigma_vert
        l_top = xsw.Layer(thickness=p.tt, sigma=sig_ox)
        l_top.add_box(-w_ox_cap / 2.0, w_ox_cap / 2.0, mat_sio2)
        if p.tc > 0:
            l_top.add_box(-w_c_cap / 2.0, -w_ox_cap / 2.0, mat_c2o)
            l_top.add_box(w_ox_cap / 2.0, w_c_cap / 2.0, mat_c2o)
        structure.add_layer(l_top)

    # 2. Upper trench slices: height (h - tg - tc)
    # Outside the line is VACUUM.
    # Inside the line is Si3N4 core flanked by SiO2 sidewalls, and outermost C2O sidewalls.
    h_upper = p.h - p.tg - p.tc
    n_upper = max(2, int(round(num_slices * (h_upper / p.h))))
    dz_upper = h_upper / n_upper

    for j in range(n_upper):
        z_local = (j + 0.5) * dz_upper
        wn = p.w + (z_local - p.h / 2.0) * (2.0 / np.tan(np.deg2rad(p.swa)))
        wn = max(1.0, min(wn, p.pitch - 2.0 * (p.tt + p.tc) - 1e-4))

        layer = xsw.Layer(thickness=dz_upper, sigma=0.0)
        # Central nitride core
        layer.add_box(-wn / 2.0, wn / 2.0, mat_si3n4, sigma=p.sigma_sw)
        # Conformal oxide sidewalls
        if p.tt > 0:
            layer.add_box(-wn / 2.0 - p.tt, -wn / 2.0, mat_sio2)
            layer.add_box(wn / 2.0, wn / 2.0 + p.tt, mat_sio2)
        # Conformal carbon sidewalls
        if p.tc > 0:
            layer.add_box(-wn / 2.0 - p.tt - p.tc, -wn / 2.0 - p.tt, mat_c2o)
            layer.add_box(wn / 2.0 + p.tt, wn / 2.0 + p.tt + p.tc, mat_c2o)
        structure.add_layer(layer)

    # 3. Trench floor carbon layer (if tc > 0)
    # Carbon coats the floor of the groove (on top of the groove oxide)
    if p.tc > 0:
        z_floor = h_upper + 0.5 * p.tc
        wn_floor = p.w + (z_floor - p.h / 2.0) * (2.0 / np.tan(np.deg2rad(p.swa)))
        wn_floor = max(1.0, min(wn_floor, p.pitch - 2.0 * p.tt - 1e-4))

        l_floor = xsw.Layer(thickness=p.tc, sigma=p.sigma_vert)
        l_floor.add_box(-wn_floor / 2.0, wn_floor / 2.0, mat_si3n4, sigma=p.sigma_sw)
        if p.tt > 0:
            l_floor.add_box(-wn_floor / 2.0 - p.tt, -wn_floor / 2.0, mat_sio2)
            l_floor.add_box(wn_floor / 2.0, wn_floor / 2.0 + p.tt, mat_sio2)
        l_floor.add_box(-p.pitch / 2.0, -wn_floor / 2.0 - p.tt, mat_c2o)
        l_floor.add_box(wn_floor / 2.0 + p.tt, p.pitch / 2.0, mat_c2o)
        structure.add_layer(l_floor)

    # 4. Lower trench slices: groove oxide (height tg)
    # Groove oxide fills the space between grating lines.
    n_lower = max(2, num_slices - n_upper - (1 if p.tc > 0 else 0))
    dz_lower = p.tg / n_lower

    for k in range(n_lower):
        z_local = h_upper + p.tc + (k + 0.5) * dz_lower
        wn = p.w + (z_local - p.h / 2.0) * (2.0 / np.tan(np.deg2rad(p.swa)))
        wn = max(1.0, min(wn, p.pitch - 2.0))

        # First lower slice interface marks the top surface of the groove oxide if no carbon
        sig = p.sigma_vert if (k == 0 and p.tc == 0.0) else 0.0
        layer = xsw.Layer(thickness=dz_lower, sigma=sig)
        # Nitride core
        layer.add_box(-wn / 2.0, wn / 2.0, mat_si3n4, sigma=p.sigma_sw)
        # Oxide extends from nitride edges to unit cell boundaries
        layer.add_box(-p.pitch / 2.0, -wn / 2.0, mat_sio2)
        layer.add_box(wn / 2.0, p.pitch / 2.0, mat_sio2)
        structure.add_layer(layer)

    # 5. Substrate native oxide layer
    if p.tsub > 0:
        l_sub = xsw.Layer(thickness=p.tsub, sigma=p.sigma_vert)
        l_sub.add_box(-p.pitch / 2.0, p.pitch / 2.0, mat_sio2)
        structure.add_layer(l_sub)

    return structure, mat_si3n4, mat_sio2, mat_c2o
