"""Visualization of multislice Structure geometry and GIXRF fit results."""
from pathlib import Path
from typing import Any, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import xsw
from .data import ExperimentData
from .geometry import build_andrle_b_sample
from .optimizer import FitResult


# Material color palette
MATERIAL_COLORS = {
    "si3n4": {"face": "#a1d99b", "edge": "#2ca02c", "label": "Si$_3$N$_4$ (core)"},
    "sio2": {"face": "#9ecae1", "edge": "#3182bd", "label": "SiO$_2$ (oxide)"},
    "c2o": {"face": "#b3b3b3", "edge": "#525252", "label": "C$_2$O (contamination)"},
    "si": {"face": "#636363", "edge": "#252525", "label": "Si (substrate)"},
    "vacuum": {"face": "#f7f7f7", "edge": "#d9d9d9", "label": "Vacuum / Air"},
}


def plot_structure(
    structure: xsw.Structure,
    ax: Optional[Axes] = None,
    *,
    substrate_depth: float = 12.0,
    show_slices: bool = True,
    title: Optional[str] = "Reconstructed Grating Cross-Section",
) -> Axes:
    """Plot the 2D cross-section of an xsw.Structure with its layers and boxes.

    Parameters
    ----------
    structure : xsw.Structure
        Grating structure assembled with Layer and Box elements.
    ax : matplotlib.axes.Axes, optional
        Target axis. If None, creates a new figure and axis.
    substrate_depth : float
        Depth of the visible substrate region below Y = 0 in nm.
    show_slices : bool
        Whether to draw horizontal slice dividing lines.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 7), layout="constrained")

    period = structure.period
    half_p = period / 2.0

    # Total height of all layers
    total_h = sum(layer.thickness for layer in structure.layers)

    # 1. Substrate patch below Y = 0
    sub_color = MATERIAL_COLORS["si"]
    rect_sub = patches.Rectangle(
        (-half_p, -substrate_depth),
        period,
        substrate_depth,
        facecolor=sub_color["face"],
        edgecolor=sub_color["edge"],
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(rect_sub)
    ax.text(0, -substrate_depth / 2.0, "Si substrate", color="white",
            ha="center", va="center", fontweight="bold", fontsize=10, zorder=3)

    # 2. Iterate through layers from top to bottom
    # Physical height Y = total_h - z (so Y = 0 at substrate top)
    legend_handles = {"Si (substrate)": rect_sub}

    current_z = 0.0
    for layer in structure.layers:
        dz = layer.thickness
        y_bottom = total_h - (current_z + dz)
        y_top = total_h - current_z

        regions = layer.regions(period)
        for r in regions:
            mat_name = r.material.name.lower()
            style = MATERIAL_COLORS.get(mat_name, {"face": "#cccccc", "edge": "#666666", "label": r.material.name})
            lbl = str(style.get("label", r.material.name))

            box_w = r.right - r.left
            rect = patches.Rectangle(
                (r.left, y_bottom),
                box_w,
                dz,
                facecolor=style["face"],
                edgecolor=style["edge"] if show_slices else "none",
                linewidth=0.7 if show_slices else 0.0,
                zorder=2,
            )
            ax.add_patch(rect)
            if lbl not in legend_handles and mat_name != "vacuum":
                legend_handles[lbl] = rect

        current_z += dz

    # Unit cell boundary guides
    ax.axvline(-half_p, color="#737373", linestyle="--", linewidth=1.0, zorder=4)
    ax.axvline(+half_p, color="#737373", linestyle="--", linewidth=1.0, zorder=4)

    # Substrate interface line
    ax.axhline(0.0, color="black", linewidth=1.5, zorder=4)

    ax.set_xlim(-half_p * 1.15, half_p * 1.15)
    ax.set_ylim(-substrate_depth * 1.1, total_h * 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel("X (nm)", fontsize=11)
    ax.set_ylabel("Height Y (nm)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    return ax


def plot_reconstruction_summary(
    data: ExperimentData,
    fit_result: FitResult,
    num_slices: int = 12,
    output_path: Optional[str | Path] = None,
    close: bool = False,
) -> Figure:
    """Generate a publication-ready figure: Channel fits and 2D reconstructed structure.

    If carbon data is present, produces a 4-panel figure (N, O, C, and Structure).
    Otherwise, produces a 3-panel figure (N, O, and Structure).

    Parameters
    ----------
    data : ExperimentData
        Experimental dataset.
    fit_result : FitResult
        Fit result object from optimizer.
    num_slices : int
        Number of slices for structure cross-section.
    output_path : str or Path, optional
        If provided, saves figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fwd = fit_result.forward
    has_c = data.has_carbon and fwd.scaled_C is not None

    if has_c:
        fig = plt.figure(figsize=(19, 4.8), layout="constrained")
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.9])
        ax_n = fig.add_subplot(gs[0, 0])
        ax_o = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_geo = fig.add_subplot(gs[0, 3])
    else:
        fig = plt.figure(figsize=(15, 5), layout="constrained")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9])
        ax_n = fig.add_subplot(gs[0, 0])
        ax_o = fig.add_subplot(gs[0, 1])
        ax_geo = fig.add_subplot(gs[0, 2])
        ax_c = None

    # 1. Nitrogen (N-Kalpha)
    ax_n.errorbar(data.angles_deg, data.exp_N, yerr=data.sigma_N, fmt="o",
                  color="#2ca02c", markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.55,
                  zorder=2, label="Experiment (N-K$\\alpha$)")
    if data.jcm_N is not None:
        ax_n.plot(data.angles_deg, data.jcm_N, "--", color="black", lw=1.5, zorder=4, label="JCMsuite Ref")
    ax_n.plot(data.angles_deg, fwd.scaled_N, "-", color="#1b7837", lw=2.2, zorder=5,
              label=f"xsw fit ($\\chi^2_N={fwd.chi2_N:.1f}$)")
    ax_n.set_xlabel("Grazing angle $\\alpha$ (deg)", fontsize=11)
    ax_n.set_ylabel("Fluorescence intensity", fontsize=11)
    ax_n.set_title("Nitrogen N-K$\\alpha$ Fit", fontsize=12, fontweight="bold")
    ax_n.legend(fontsize=8.5)
    ax_n.grid(True, linestyle=":", alpha=0.5)

    # 2. Oxygen (O-Kalpha)
    ax_o.errorbar(data.angles_deg, data.exp_O, yerr=data.sigma_O, fmt="s",
                  color="#1f78b4", markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.55,
                  zorder=2, label="Experiment (O-K$\\alpha$)")
    if data.jcm_O is not None:
        ax_o.plot(data.angles_deg, data.jcm_O, "--", color="black", lw=1.5, zorder=4, label="JCMsuite Ref")
    ax_o.plot(data.angles_deg, fwd.scaled_O, "-", color="#08519c", lw=2.2, zorder=5,
              label=f"xsw fit ($\\chi^2_O={fwd.chi2_O:.1f}$)")
    ax_o.set_xlabel("Grazing angle $\\alpha$ (deg)", fontsize=11)
    ax_o.set_ylabel("Fluorescence intensity", fontsize=11)
    ax_o.set_title("Oxygen O-K$\\alpha$ Fit", fontsize=12, fontweight="bold")
    ax_o.legend(fontsize=8.5)
    ax_o.grid(True, linestyle=":", alpha=0.5)

    # 3. Carbon (C-Kalpha) if available
    if has_c and ax_c is not None:
        ax_c.errorbar(data.angles_deg, data.exp_C, yerr=data.sigma_C, fmt="^",
                      color="#636363", markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.55,
                      zorder=2, label="Experiment (C-K$\\alpha$)")
        if data.jcm_C is not None:
            ax_c.plot(data.angles_deg, data.jcm_C, "--", color="black", lw=1.5, zorder=4, label="JCMsuite Ref")
        ax_c.plot(data.angles_deg, fwd.scaled_C, "-", color="#252525", lw=2.2, zorder=5,
                  label=f"xsw fit ($\\chi^2_C={fwd.chi2_C:.1f}$)")
        ax_c.set_xlabel("Grazing angle $\\alpha$ (deg)", fontsize=11)
        ax_c.set_ylabel("Fluorescence intensity", fontsize=11)
        ax_c.set_title("Carbon C-K$\\alpha$ Fit", fontsize=12, fontweight="bold")
        ax_c.legend(fontsize=8.5)
        ax_c.grid(True, linestyle=":", alpha=0.5)

    # 4. Reconstructed 2D Structure
    struct, _, _, _ = build_andrle_b_sample(fit_result.params, num_slices=num_slices)
    plot_structure(
        struct,
        ax=ax_geo,
        title=f"Reconstructed Profile\nh={fit_result.params.h:.1f}nm, w={fit_result.params.w:.1f}nm, $t_C$={fit_result.params.tc:.1f}nm",
    )

    fig.suptitle(
        f"GIXRF Reconstruction (680 eV) — Reduced $\\chi^2/\\mathrm{{DoF}} = {fit_result.reduced_chi2:.2f}$",
        fontsize=13,
        fontweight="bold",
    )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        if close:
            plt.close(fig)

    return fig


def plot_reconstruction_single_graph(
    data: ExperimentData,
    fit_result: FitResult,
    output_path: Optional[str | Path] = None,
    close: bool = False,
) -> Figure:
    """Plot all experimental fluorescence channels and model fits on a single graph (matching Figure 4.37).

    Parameters
    ----------
    data : ExperimentData
        Experimental dataset.
    fit_result : FitResult
        Fit result object from optimizer.
    output_path : str or Path, optional
        If provided, saves figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    fwd = fit_result.forward

    # Oxygen: Blue
    ax.errorbar(data.angles_deg, data.exp_O, yerr=data.sigma_O, fmt="s",
                color="#3182bd", markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.5,
                zorder=2, label=f"O-K$\\alpha$ Exp")
    ax.plot(data.angles_deg, fwd.scaled_O, "-", color="#08519c", lw=2.2, zorder=5,
            label=f"O-K$\\alpha$ Fit ($\\chi^2_O={fwd.chi2_O:.1f}$)")

    # Nitrogen: Green
    ax.errorbar(data.angles_deg, data.exp_N, yerr=data.sigma_N, fmt="o",
                color="#31a354", markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.5,
                zorder=2, label=f"N-K$\\alpha$ Exp")
    ax.plot(data.angles_deg, fwd.scaled_N, "-", color="#006d2c", lw=2.2, zorder=5,
            label=f"N-K$\\alpha$ Fit ($\\chi^2_N={fwd.chi2_N:.1f}$)")

    # Carbon: Grey / Black (if present)
    if data.has_carbon and fwd.scaled_C is not None:
        ax.errorbar(data.angles_deg, data.exp_C, yerr=data.sigma_C, fmt="^",
                    color="#737373", markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.5,
                    zorder=2, label=f"C-K$\\alpha$ Exp")
        ax.plot(data.angles_deg, fwd.scaled_C, "-", color="#111111", lw=2.2, zorder=5,
                label=f"C-K$\\alpha$ Fit ($\\chi^2_C={fwd.chi2_C:.1f}$)")

    ax.set_xlabel("Grazing angle of incidence $\\theta$ (deg)", fontsize=12)
    ax.set_ylabel("Fluorescence intensity (counts)", fontsize=12)
    ax.set_title(
        f"GIXRF Fluorescence vs Angle (Anna Andrle Fig. 4.37)\n"
        f"Reduced $\\chi^2/\\mathrm{{DoF}} = {fit_result.reduced_chi2:.2f}$ (E = 680 eV, p-pol)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        if close:
            plt.close(fig)

    return fig
