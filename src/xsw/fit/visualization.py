"""Visualization of multislice Structure geometry and GIXRF fit results."""
from pathlib import Path
from typing import Optional
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import ScaledTranslation
import numpy as np

import xsw
from .data import ExperimentData
from .geometry import build_andrle_b_sample
from .optimizer import FitResult


# Material color palette
MATERIAL_COLORS = {
    "si3n4": {"face": "#a1d99b", "edge": "#2ca02c", "en": "Si$_3$N$_4$ (core)", "ru": "Si$_3$N$_4$ (сердцевина)"},
    "sio2": {"face": "#9ecae1", "edge": "#3182bd", "en": "SiO$_2$ (oxide)", "ru": "SiO$_2$ (оксид)"},
    "c2o": {"face": "#b3b3b3", "edge": "#525252", "en": "C$_2$O (contamination)", "ru": "C$_2$O (загрязнение)"},
    "si": {"face": "#636363", "edge": "#252525", "en": "Si (substrate)", "ru": "Si (подложка)"},
    "vacuum": {"face": "#f7f7f7", "edge": "#d9d9d9", "en": "Vacuum / Air", "ru": "Вакуум / воздух"},
}

_UI_TEXT = {
    "en": {
        "struct_title": "Reconstructed Grating Cross-Section",
        "substrate_text": "Si substrate",
        "x_nm": "X (nm)",
        "y_nm": "Height Y (nm)",
        "profile_title": "Reconstructed Profile\nh={h:.1f}nm, w={w:.1f}nm, $t_C$={tc:.1f}nm",
        "angle_alpha": "Grazing angle $\\alpha$ (deg)",
        "angle_theta": "Grazing angle of incidence $\\theta$ (deg)",
        "intensity": "Fluorescence intensity",
        "intensity_counts": "Fluorescence intensity (counts)",
        "fit_suffix": "Fit",
        "exp_suffix": "Exp",
        "exp_paren": "Experiment ({name})",
        "fit_chi2": "xsw fit ($\\chi^2_{{{el}}}={chi2:.1f}$)",
        "summary_suptitle": "GIXRF Reconstruction (680 eV) — Reduced $\\chi^2/\\mathrm{{DoF}} = {rchi2:.2f}$",
        "single_title": (
            "GIXRF Fluorescence vs Angle (Anna Andrle Fig. 4.37)\n"
            "Reduced $\\chi^2/\\mathrm{{DoF}} = {rchi2:.2f}$ (E = 680 eV, p-pol)"
        ),
        "n_fit": "Nitrogen N-K$\\alpha$ Fit",
        "o_fit": "Oxygen O-K$\\alpha$ Fit",
        "c_fit": "Carbon C-K$\\alpha$ Fit",
        "jcm": "JCMsuite Ref",
        "heatmap_cb": "Fluorescence intensity / incident photon",
    },
    "ru": {
        "struct_title": "Восстановленный профиль решётки",
        "substrate_text": "Подложка Si",
        "x_nm": "X (нм)",
        "y_nm": "Высота Y (нм)",
        "profile_title": "Восстановленный профиль\nh = {h:.1f} нм, w = {w:.1f} нм, $t_C$ = {tc:.1f} нм",
        "angle_alpha": "Угол скользящего падения $\\alpha$ (град)",
        "angle_theta": "Угол скользящего падения $\\theta$ (град)",
        "intensity": "Интенсивность флуоресценции (отн. ед.)",
        "intensity_counts": "Интенсивность флуоресценции (отн. ед.)",
        "fit_suffix": "подгонка",
        "exp_suffix": "эксперимент",
        "exp_paren": "Эксперимент ({name})",
        "fit_chi2": "Расчёт xsw ($\\chi^2_{{{el}}}={chi2:.1f}$)",
        "summary_suptitle": "Реконструкция GIXRF (680 эВ) — $\\chi^2/\\mathrm{{DoF}} = {rchi2:.2f}$",
        "single_title": (
            "Интенсивность флуоресценции GIXRF (рис. 4.37 А. Андрле)\n"
            "$\\chi^2/\\mathrm{{DoF}} = {rchi2:.2f}$ ($E = 680$ эВ, $p$-поляризация)"
        ),
        "n_fit": "Подгонка азота N-K$\\alpha$",
        "o_fit": "Подгонка кислорода O-K$\\alpha$",
        "c_fit": "Подгонка углерода C-K$\\alpha$",
        "jcm": "Эталон JCMsuite",
        "heatmap_cb": "Интенсивность флуоресценции / падающий фотон",
    },
}


def marker_display_indices(
    angles_deg: np.ndarray,
    dense_threshold: float = 6.0,
    dense_step: int = 4,
) -> np.ndarray:
    """Select well-spaced marker indices in dense low-angle scans while keeping sparse angles."""
    dense_mask = angles_deg < dense_threshold
    dense_idx = np.where(dense_mask)[0][::dense_step]
    sparse_idx = np.where(~dense_mask)[0]
    return np.concatenate([dense_idx, sparse_idx])


def plot_distinct_fit_and_markers(
    ax: Axes,
    angles_deg: np.ndarray,
    exp: np.ndarray,
    sigma: np.ndarray,
    fit_curve: np.ndarray,
    *,
    fmt: str,
    fit_color: str,
    fill_color: str,
    edge_color: str,
    exp_label: str,
    fit_label: str,
    dense_step: int = 4,
) -> None:
    """Plot experimental markers with contrasting shades/shadows and white-haloed fit curve."""
    idx = marker_display_indices(angles_deg, dense_step=dense_step)
    shadow_trans = ax.transData + ScaledTranslation(0.9 / 72.0, -0.9 / 72.0, ax.figure.dpi_scale_trans)

    # Marker drop shadow in point coordinates so it scales uniformly across axes
    ax.plot(
        angles_deg[idx], exp[idx], fmt,
        color="black", markersize=5.4, alpha=0.18,
        transform=shadow_trans, zorder=2, label="_nolegend_",
    )
    ax.errorbar(
        angles_deg[idx], exp[idx], yerr=sigma[idx], fmt=fmt,
        color=edge_color, mfc=fill_color, mec=edge_color, mew=1.15, markersize=5.1,
        elinewidth=0.9, capsize=1.8, alpha=0.95, zorder=3, label=exp_label,
    )

    # Fit curve soft drop shadow + crisp white halo on top for clean separation
    ax.plot(
        angles_deg, fit_curve, "-",
        color="black", lw=2.5, alpha=0.14,
        transform=shadow_trans, zorder=4, label="_nolegend_",
    )
    line, = ax.plot(
        angles_deg, fit_curve, "-",
        color=fit_color, lw=2.1, zorder=5, label=fit_label,
    )
    line.set_path_effects([pe.withStroke(linewidth=3.4, foreground="white", alpha=0.90), pe.Normal()])


def _get_labels(lang: str) -> dict[str, str]:
    if lang not in _UI_TEXT:
        raise ValueError(f"Unsupported language '{lang}'. Choose from {tuple(_UI_TEXT)}.")
    return _UI_TEXT[lang]


def _save_figure(fig: Figure, output_path: Optional[str | Path], dpi: int = 450, close: bool = False) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        if close:
            plt.close(fig)


def plot_structure(
    structure: xsw.Structure,
    ax: Optional[Axes] = None,
    *,
    substrate_depth: float = 12.0,
    show_slices: bool = True,
    title: Optional[str] = None,
    lang: str = "en",
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
        Plot title. Defaults to a localized title when omitted.
    lang : {'en', 'ru'}, default 'en'
        Label language for axes, legend, and substrate annotation.

    Returns
    -------
    matplotlib.axes.Axes
    """
    txt = _get_labels(lang)
    if title is None:
        title = txt["struct_title"]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 7), layout="constrained")

    period = structure.period
    half_p = period / 2.0
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
    ax.text(0, -substrate_depth / 2.0, txt["substrate_text"], color="white",
            ha="center", va="center", fontweight="bold", fontsize=10, zorder=3)

    # 2. Iterate through layers from top to bottom (Y = 0 at substrate top)
    legend_handles = {sub_color[lang]: rect_sub}

    current_z = 0.0
    for layer in structure.layers:
        dz = layer.thickness
        y_bottom = total_h - (current_z + dz)

        for r in layer.regions(period):
            mat_name = r.material.name.lower()
            style = MATERIAL_COLORS.get(
                mat_name,
                {"face": "#cccccc", "edge": "#666666", lang: r.material.name},
            )
            lbl = str(style.get(lang, r.material.name))

            rect = patches.Rectangle(
                (r.left, y_bottom),
                r.right - r.left,
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

    ax.axvline(-half_p, color="#737373", linestyle="--", linewidth=1.0, zorder=4)
    ax.axvline(+half_p, color="#737373", linestyle="--", linewidth=1.0, zorder=4)
    ax.axhline(0.0, color="black", linewidth=1.5, zorder=4)

    ax.set_xlim(-half_p * 1.15, half_p * 1.15)
    ax.set_ylim(-substrate_depth * 1.1, total_h * 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel(txt["x_nm"], fontsize=11)
    ax.set_ylabel(txt["y_nm"], fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="upper right",
        framealpha=0.92,
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
    lang: str = "en",
) -> Figure:
    """Generate a publication-ready figure: Channel fits and 2D reconstructed structure."""
    txt = _get_labels(lang)
    fwd = fit_result.forward
    has_c = data.has_carbon and fwd.scaled_C is not None

    if has_c:
        fig = plt.figure(figsize=(19, 4.8), layout="constrained")
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.9])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
        ax_geo = fig.add_subplot(gs[0, 3])
    else:
        fig = plt.figure(figsize=(15, 5), layout="constrained")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        ax_geo = fig.add_subplot(gs[0, 2])

    channels = [
        ("N", "N-K$\\alpha$", txt["n_fit"], data.exp_N, data.sigma_N, fwd.scaled_N, fwd.chi2_N,
         data.jcm_N, "o", "#006d2c", "#a1d99b", "#238b45"),
        ("O", "O-K$\\alpha$", txt["o_fit"], data.exp_O, data.sigma_O, fwd.scaled_O, fwd.chi2_O,
         data.jcm_O, "s", "#08519c", "#9ecae1", "#2171b5"),
    ]
    if has_c and data.exp_C is not None and data.sigma_C is not None and fwd.chi2_C is not None:
        channels.append(
            ("C", "C-K$\\alpha$", txt["c_fit"], data.exp_C, data.sigma_C, fwd.scaled_C, fwd.chi2_C,
             data.jcm_C, "^", "#111111", "#d9d9d9", "#525252")
        )

    for ax, (el, sym, panel_title, exp, sig, sc, chi2, jcm, fmt, c_fit, c_fill, c_edge) in zip(axes, channels):
        if jcm is not None:
            ax.plot(data.angles_deg, jcm, "--", color="black", lw=1.4, zorder=4, label=txt["jcm"])
        plot_distinct_fit_and_markers(
            ax, data.angles_deg, exp, sig, sc,
            fmt=fmt, fit_color=c_fit, fill_color=c_fill, edge_color=c_edge,
            exp_label=txt["exp_paren"].format(name=sym),
            fit_label=txt["fit_chi2"].format(el=el, chi2=chi2),
        )
        ax.set_xlabel(txt["angle_alpha"], fontsize=11)
        ax.set_ylabel(txt["intensity"], fontsize=11)
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, framealpha=0.92)
        ax.grid(True, linestyle=":", alpha=0.5)

    struct, _, _, _ = build_andrle_b_sample(fit_result.params, num_slices=num_slices)
    plot_structure(
        struct,
        ax=ax_geo,
        title=txt["profile_title"].format(
            h=fit_result.params.h, w=fit_result.params.w, tc=fit_result.params.tc
        ),
        lang=lang,
    )

    fig.suptitle(txt["summary_suptitle"].format(rchi2=fit_result.reduced_chi2), fontsize=13, fontweight="bold")
    _save_figure(fig, output_path, dpi=450, close=close)
    return fig


def plot_reconstruction_single_graph(
    data: ExperimentData,
    fit_result: FitResult,
    output_path: Optional[str | Path] = None,
    close: bool = False,
    lang: str = "en",
) -> Figure:
    """Plot all experimental fluorescence channels and model fits on a single graph (matching Figure 4.37)."""
    txt = _get_labels(lang)
    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    fwd = fit_result.forward

    exp_lbl = txt["exp_suffix"]
    fit_lbl = txt["fit_suffix"]

    plot_distinct_fit_and_markers(
        ax, data.angles_deg, data.exp_O, data.sigma_O, fwd.scaled_O,
        fmt="s", fit_color="#08519c", fill_color="#9ecae1", edge_color="#2171b5",
        exp_label=f"O-K$\\alpha$ — {exp_lbl}",
        fit_label=f"O-K$\\alpha$ — {fit_lbl} ($\\chi^2_O={fwd.chi2_O:.1f}$)",
    )
    plot_distinct_fit_and_markers(
        ax, data.angles_deg, data.exp_N, data.sigma_N, fwd.scaled_N,
        fmt="o", fit_color="#006d2c", fill_color="#a1d99b", edge_color="#238b45",
        exp_label=f"N-K$\\alpha$ — {exp_lbl}",
        fit_label=f"N-K$\\alpha$ — {fit_lbl} ($\\chi^2_N={fwd.chi2_N:.1f}$)",
    )
    if data.has_carbon and data.exp_C is not None and data.sigma_C is not None and fwd.scaled_C is not None:
        plot_distinct_fit_and_markers(
            ax, data.angles_deg, data.exp_C, data.sigma_C, fwd.scaled_C,
            fmt="^", fit_color="#111111", fill_color="#d9d9d9", edge_color="#525252",
            exp_label=f"C-K$\\alpha$ — {exp_lbl}",
            fit_label=f"C-K$\\alpha$ — {fit_lbl} ($\\chi^2_C={fwd.chi2_C:.1f}$)",
        )

    ax.set_xlabel(txt["angle_theta"], fontsize=12)
    ax.set_ylabel(txt["intensity_counts"], fontsize=12)
    ax.set_title(txt["single_title"].format(rchi2=fit_result.reduced_chi2), fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.5)

    _save_figure(fig, output_path, dpi=450, close=close)
    return fig


def plot_andrle_heatmap(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    int_N: np.ndarray,
    int_O: np.ndarray,
    *,
    vmax: float = 8.5e-5,
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    close: bool = False,
    lang: str = "en",
) -> Figure:
    """Plot 2D angular fluorescence heatmaps replicating Figure 4.10 from Anna Andrle's thesis."""
    txt = _get_labels(lang)
    fig = plt.figure(figsize=(9, 4.8), dpi=450)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.0, 1.0, 0.045, 0.045],
        wspace=0.035,
        left=0.08, right=0.88, top=0.92, bottom=0.15,
    )

    ax_n = fig.add_subplot(gs[0, 0])
    ax_o = fig.add_subplot(gs[0, 1], sharey=ax_n)
    ax_cb_n = fig.add_subplot(gs[0, 2])
    ax_cb_o = fig.add_subplot(gs[0, 3])

    norm = Normalize(vmin=0.0, vmax=vmax)

    im_n = ax_n.imshow(
        np.fliplr(int_N),
        origin="lower",
        extent=[-phi_deg[-1], 0.0, theta_deg[0], theta_deg[-1]],
        aspect="auto",
        cmap="Greens_r",
        norm=norm,
        interpolation="bicubic",
    )

    im_o = ax_o.imshow(
        int_O,
        origin="lower",
        extent=[0.0, phi_deg[-1], theta_deg[0], theta_deg[-1]],
        aspect="auto",
        cmap="Blues_r",
        norm=norm,
        interpolation="bicubic",
    )

    ax_n.set_xlim(-phi_deg[-1], 0.0)
    ax_n.set_ylim(theta_deg[0], 6.0)
    ax_o.set_xlim(0.0, phi_deg[-1])
    ax_o.set_ylim(theta_deg[0], 6.0)

    ax_n.set_xticks([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5])
    ax_n.set_yticks([1, 2, 3, 4, 5, 6])
    ax_o.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    plt.setp(ax_o.get_yticklabels(), visible=False)

    ax_n.tick_params(direction="out", length=4)
    ax_o.tick_params(direction="out", length=4)

    ax_n.set_xlabel(r"$- \varphi \;/\; ^\circ$", fontsize=13)
    ax_o.set_xlabel(r"$\varphi \;/\; ^\circ$", fontsize=13)
    ax_n.set_ylabel(r"$\theta \;/\; ^\circ$", fontsize=13)

    badge_n = dict(boxstyle="round,pad=0.35,rounding_size=0.4", fc="#d9f0d3", ec="#1b7837", lw=2.2)
    ax_n.text(0.08, 0.92, "N-Kα", transform=ax_n.transAxes,
              fontsize=12, va="top", ha="left", bbox=badge_n, fontweight="bold")

    badge_o = dict(boxstyle="round,pad=0.35,rounding_size=0.4", fc="#deebf7", ec="#2171b5", lw=2.2)
    ax_o.text(0.08, 0.92, "O-Kα", transform=ax_o.transAxes,
              fontsize=12, va="top", ha="left", bbox=badge_o, fontweight="bold")

    fig.colorbar(im_n, cax=ax_cb_n)
    cb_o = fig.colorbar(im_o, cax=ax_cb_o)

    ax_cb_n.set_yticks([])
    ax_cb_n.tick_params(length=0)

    cb_ticks = np.linspace(0.0, 8.0e-5, 9)
    cb_o.set_ticks(cb_ticks)
    ax_cb_o.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
    ax_cb_o.tick_params(left=True, right=True, direction="out", length=3.5)
    ax_cb_o.set_ylabel(txt["heatmap_cb"], fontsize=11.5, labelpad=10)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    _save_figure(fig, output_path, dpi=450, close=close)
    return fig
