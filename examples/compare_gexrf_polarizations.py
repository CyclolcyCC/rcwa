"""Comparison of s vs p polarization in Grazing Emission (GEXRF).

Replicates the polarization comparison graph (Is vs Ip) for the Grazing Emission
geometry using the optical reciprocity theorem for the reconstructed nanograting model.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import numpy as np

# Ensure xsw is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import xsw
import xsw.fit as fit

# Global worker state for multi-core solves
_worker_prep = None
_worker_mat_target = None
_worker_lam_em = None


def _init_worker(params_dict, E_em, mat_key, max_order):
    global _worker_prep, _worker_mat_target, _worker_lam_em
    p = fit.AndrleBParams(**params_dict)
    _worker_lam_em = 1239.841984 / E_em
    chi_si = xsw.chi_from_formula("Si", density=2.33, energy=E_em)
    chi_n = xsw.chi_from_formula("Si3N4", density=3.10, energy=E_em)
    chi_o = xsw.chi_from_formula("SiO2", density=2.20, energy=E_em)
    chi_c = xsw.chi_from_formula("C2O", density=1.80, energy=E_em)
    struct, mat_n, mat_o, mat_c = fit.build_andrle_b_sample(
        p, num_slices=10, optical_constants=(chi_si, chi_n, chi_o, chi_c)
    )
    _worker_prep = xsw.prepare(struct, max_order=max_order)
    _worker_mat_target = {"Si3N4": mat_n, "SiO2": mat_o, "C2O": mat_c}[mat_key]


def _calc_batch(betas_rad):
    res = []
    for b in betas_rad:
        sol_s = _worker_prep.solve(alpha=b, phi=0.0, wavelength=_worker_lam_em, polarization="s")
        sol_p = _worker_prep.solve(alpha=b, phi=0.0, wavelength=_worker_lam_em, polarization="p")
        res.append((
            sol_s.fluorescence(_worker_mat_target, attenuation=1e-3),
            sol_p.fluorescence(_worker_mat_target, attenuation=1e-3),
        ))
    return res


def _solve_angles(betas_deg, params_dict, E_em, mat_key, max_order=12):
    betas_rad = np.deg2rad(betas_deg)
    chunk_size = max(1, (len(betas_rad) + 7) // 8)
    chunks = [betas_rad[i:i + chunk_size] for i in range(0, len(betas_rad), chunk_size)]
    with ProcessPoolExecutor(
        max_workers=min(8, len(chunks)),
        initializer=_init_worker,
        initargs=(params_dict, E_em, mat_key, max_order),
    ) as pool:
        flat = [item for sub in pool.map(_calc_batch, chunks) for item in sub]
    return np.array([r[0] for r in flat]), np.array([r[1] for r in flat])


def _load_params_dict(fit_file):
    result = fit.FitResult.load(fit_file)
    p = result.params
    return p, {
        k: getattr(p, k)
        for k in ["h", "w", "swa", "tt", "tg", "sigma_sw", "sigma_vert", "tc", "tsub", "pitch"]
    }


def main():
    output_dir = REPO_ROOT / "examples" / "output"
    data_file = output_dir / "gexrf_sp_data.npz"
    fit_file = output_dir / "fit_andrle_b_results.npz"

    # %% 1. Load or Compute GEXRF s vs p Curves
    # Characteristic line: N-Kalpha (392.4 eV) from central Si3N4 grating core
    # (Matches the intensity scale [0, 2500] in the reference comparison graph)
    E_em = 392.4
    mat_key = "Si3N4"
    zoom_min, zoom_max = 16.00, 16.10

    if data_file.exists() and "--recalc" not in sys.argv:
        print(f"Loading pre-calculated GEXRF s/p data from {data_file.name}...")
        with np.load(data_file) as data:
            betas_deg, Is, Ip = data["betas_deg"], data["Is"], data["Ip"]
            has_zoom = (
                "zoom_betas_deg" in data
                and np.isclose(data["zoom_betas_deg"][0], zoom_min)
                and np.isclose(data["zoom_betas_deg"][-1], zoom_max)
            )
            if has_zoom:
                zoom_betas_deg = data["zoom_betas_deg"]
                zoom_Is = data["zoom_Is"]
                zoom_Ip = data["zoom_Ip"]
        if not has_zoom:
            _, params_dict = _load_params_dict(fit_file)
            zoom_betas_deg = np.linspace(zoom_min, zoom_max, 21)
            print(f"Computing high-resolution inset scan [{zoom_min:.2f}°, {zoom_max:.2f}°]...")
            zoom_Is, zoom_Ip = _solve_angles(zoom_betas_deg, params_dict, E_em, mat_key)
            np.savez_compressed(
                data_file, betas_deg=betas_deg, Is=Is, Ip=Ip,
                zoom_betas_deg=zoom_betas_deg, zoom_Is=zoom_Is, zoom_Ip=zoom_Ip,
            )
    else:
        p, params_dict = _load_params_dict(fit_file)
        betas_deg = np.linspace(0.05, 20.0, 200)
        zoom_betas_deg = np.linspace(zoom_min, zoom_max, 21)

        print(f"Reconstructed Grating: h={p.h:.1f}nm, w={p.w:.1f}nm, swa={p.swa:.1f}°, tc={p.tc:.1f}nm")
        print(f"Computing GEXRF s vs p curves across {len(betas_deg)} emission angles (N-Kalpha, 392.4 eV)...")

        t0 = time.perf_counter()
        Is, Ip = _solve_angles(betas_deg, params_dict, E_em, mat_key)
        zoom_Is, zoom_Ip = _solve_angles(zoom_betas_deg, params_dict, E_em, mat_key)
        print(f"Calculation completed in {time.perf_counter() - t0:.2f} s.")

        np.savez_compressed(
            data_file, betas_deg=betas_deg, Is=Is, Ip=Ip,
            zoom_betas_deg=zoom_betas_deg, zoom_Is=zoom_Is, zoom_Ip=zoom_Ip,
        )

    # %% 2. Plot Single Comparison Graph with Deep Inset Zoom (16.00° - 16.10°) and Delta % Arrow
    # Color scheme aligned with GIXRF/GEXRF comparison (blue, green, black)
    color_s = "#08519c"
    color_p = "#31a354"
    color_arrow = "#252525"

    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=450)

    ax.plot(betas_deg, Is, color=color_s, lw=2.5, label=r"$s$-поляризация ($I_s$)")
    ax.plot(betas_deg, Ip, color=color_p, lw=2.1, ls="--", label=r"$p$-поляризация ($I_p$)")

    ax.set_title(r"Сравнение $I_s(\beta)$ и $I_p(\beta)$", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(r"Угол выхода $\beta$, град", fontsize=13)
    ax.set_ylabel(r"Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.", fontsize=13)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, max(Is.max(), Ip.max()) * 1.06)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", linestyle="-", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=11.5)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.92, facecolor="white")

    # Inset zoom into [16.00, 16.10] deg where |Is - Ip| peaks (~2.30 rel. units)
    ax_ins = ax.inset_axes([0.08, 0.57, 0.38, 0.37])
    ax_ins.plot(zoom_betas_deg, zoom_Is, color=color_s, lw=2.6)
    ax_ins.plot(zoom_betas_deg, zoom_Ip, color=color_p, lw=2.2, ls="--")

    y_min = min(zoom_Is.min(), zoom_Ip.min())
    y_max = max(zoom_Is.max(), zoom_Ip.max())
    y_pad = (y_max - y_min) * 0.08

    ax_ins.set_xlim(zoom_min, zoom_max)
    ax_ins.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_ins.set_xticks([16.00, 16.02, 16.04, 16.06, 16.08, 16.10])
    ax_ins.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_ins.tick_params(axis="both", which="major", labelsize=9.5)
    ax_ins.grid(True, which="major", linestyle=":", alpha=0.7)
    ax_ins.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_ins.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Double-headed arrow inside inset highlighting Delta in %
    mid = len(zoom_betas_deg) // 2
    xm = float(zoom_betas_deg[mid])
    ys_m = float(zoom_Is[mid])
    yp_m = float(zoom_Ip[mid])
    delta_pct = (ys_m - yp_m) / ys_m * 100.0

    ax_ins.plot([xm - 0.004, xm + 0.004], [ys_m, ys_m], color=color_arrow, lw=0.9, ls=":")
    ax_ins.plot([xm - 0.004, xm + 0.004], [yp_m, yp_m], color=color_arrow, lw=0.9, ls=":")
    ax_ins.annotate(
        "",
        xy=(xm, ys_m),
        xytext=(xm, yp_m),
        arrowprops=dict(arrowstyle="<->", color=color_arrow, lw=1.7, shrinkA=0, shrinkB=0),
        zorder=8,
    )
    ax_ins.annotate(
        rf"$\Delta = {delta_pct:.2f}\%$",
        xy=(xm, 0.5 * (ys_m + yp_m)),
        xytext=(xm + 0.012, yp_m - 1.55),
        fontsize=10.5,
        fontweight="bold",
        color=color_arrow,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color_arrow, lw=1.0, alpha=0.96),
        arrowprops=dict(arrowstyle="-|>", color=color_arrow, lw=1.0, connectionstyle="arc3,rad=-0.15"),
        zorder=9,
    )

    # Connector lines and rectangle on main plot
    ax.indicate_inset_zoom(ax_ins, edgecolor="black", alpha=0.65, lw=1.0)

    fig.tight_layout()

    out_file = output_dir / "gexrf_sp_comparison.png"
    out_svg = output_dir / "gexrf_sp_comparison.svg"
    fig.savefig(out_file, dpi=450)
    fig.savefig(out_svg)
    print(f"Saved GEXRF s vs p comparison plot to: {out_file} and {out_svg}")

    # %% 3. Two-Panel Plot with Relative Polarization Difference
    fig_diff, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=450)

    ax1.plot(betas_deg, Is, color=color_s, lw=2.5, label=r"$s$-поляризация ($I_s$)")
    ax1.plot(betas_deg, Ip, color=color_p, lw=2.1, ls="--", label=r"$p$-поляризация ($I_p$)")
    ax1.set_title(r"Сравнение $I_s(\beta)$ и $I_p(\beta)$ (скользящий выход)", fontsize=13, fontweight="bold")
    ax1.set_xlabel(r"Угол выхода $\beta$, град", fontsize=12)
    ax1.set_ylabel(r"Интенсивность $I_{\mathrm{fluor}}$, отн. ед.", fontsize=12)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, max(Is.max(), Ip.max()) * 1.06)
    ax1.grid(True, linestyle="-", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.92)

    diff_rel = (Is - Ip) / np.maximum(Is, 1e-12) * 100.0
    ax2.plot(betas_deg, diff_rel, color=color_arrow, lw=2.0, label=r"Разность $(I_s - I_p)/I_s$")
    ax2.set_title(r"Относительная разность поляризаций", fontsize=13, fontweight="bold")
    ax2.set_xlabel(r"Угол выхода $\beta$, град", fontsize=12)
    ax2.set_ylabel(r"Разность $\delta$, %", fontsize=12)
    ax2.set_xlim(0, 20)
    ax2.grid(True, linestyle="-", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=11, framealpha=0.92)

    fig_diff.tight_layout()
    out_diff = output_dir / "gexrf_sp_comparison_with_diff.png"
    out_diff_svg = output_dir / "gexrf_sp_comparison_with_diff.svg"
    fig_diff.savefig(out_diff, dpi=450)
    fig_diff.savefig(out_diff_svg)
    print(f"Saved two-panel comparison with difference to: {out_diff} and {out_diff_svg}")

    plt.show()


if __name__ == "__main__":
    main()
