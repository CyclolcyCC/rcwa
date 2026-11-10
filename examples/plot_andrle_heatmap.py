"""Fluorescence Heatmaps (Figure 4.10 from Anna Andrle Thesis).

Calculates and displays the 2D angle-dependent fluorescence signal for
Nitrogen (N-Kalpha, green) and Oxygen (O-Kalpha, blue) across incidence
angles theta in [0.1, 6.0] deg and azimuthal angles phi in [0, 3] deg
at 680 eV (p-pol) for the reconstructed nanograting model.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np

# Ensure xsw is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import xsw
import xsw.fit as fit

# Global worker state for multi-core RCWA scans
_worker_prep = None
_worker_mat_n = None
_worker_mat_o = None


def _init_worker(params_dict, max_order):
    global _worker_prep, _worker_mat_n, _worker_mat_o
    p = fit.AndrleBParams(**params_dict)
    struct, mat_n, mat_o, _ = fit.build_andrle_b_sample(p, num_slices=10)
    _worker_prep = xsw.prepare(struct, max_order=max_order)
    _worker_mat_n = mat_n
    _worker_mat_o = mat_o


def _calc_batch(angles_deg):
    results = []
    for th_deg, ph_deg in angles_deg:
        sol = _worker_prep.solve(
            alpha=np.deg2rad(th_deg),
            phi=np.deg2rad(ph_deg),
            wavelength=1.8233,
            polarization="p",
        )
        results.append((
            sol.fluorescence(_worker_mat_n, attenuation=2.75e-3),
            sol.fluorescence(_worker_mat_o, attenuation=2.33e-3),
        ))
    return results


def main():
    output_dir = REPO_ROOT / "examples" / "output"
    data_file = output_dir / "andrle_heatmap_data.npz"

    # %% 1. Load or Compute 2D Angular Fluorescence Grid
    if data_file.exists() and "--recalc" not in sys.argv:
        print(f"Loading pre-calculated 2D heatmap data from {data_file.name}...")
        data = np.load(data_file)
        theta_deg = data["theta_deg"]
        phi_deg = data["phi_deg"]
        int_N = data["int_N"]
        int_O = data["int_O"]
    else:
        fit_file = output_dir / "fit_andrle_b_results.npz"
        result = fit.FitResult.load(fit_file)
        p = result.params
        params_dict = {k: getattr(p, k) for k in ["h", "w", "swa", "tt", "tg", "sigma_sw", "sigma_vert", "tc", "tsub", "pitch"]}

        # Grid definition matching Figure 4.10 (80 theta x 51 phi = 4080 points)
        theta_deg = np.linspace(0.1, 6.0, 80)
        phi_deg = np.linspace(0.0, 3.0, 51)
        points = [(t, ph) for t in theta_deg for ph in phi_deg]

        print(f"Calculating {len(points)} grid points for reconstructed model (h={p.h:.1f}nm, w={p.w:.1f}nm)...")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=8, initializer=_init_worker, initargs=(params_dict, 10)) as pool:
            chunks = [points[i:i + 50] for i in range(0, len(points), 50)]
            flat = [item for sub in pool.map(_calc_batch, chunks) for item in sub]

        raw_N = np.array([r[0] for r in flat]).reshape(len(theta_deg), len(phi_deg))
        raw_O = np.array([r[1] for r in flat]).reshape(len(theta_deg), len(phi_deg))
        print(f"Calculation finished in {time.perf_counter() - t0:.1f} s.")

        # Match "Fluorescence intensity / incident photon" scale from thesis BestFit
        bf = np.load(REPO_ROOT / "legacy" / "self_attenuation" / "20200417_BestFit.npy")
        scale_n = np.sum(np.interp(theta_deg, bf[0], bf[3]) * raw_N[:, 0]) / np.sum(raw_N[:, 0]**2)
        scale_o = np.sum(np.interp(theta_deg, bf[0], bf[4]) * raw_O[:, 0]) / np.sum(raw_O[:, 0]**2)

        int_N = raw_N * scale_n
        int_O = raw_O * scale_o
        np.savez_compressed(data_file, theta_deg=theta_deg, phi_deg=phi_deg, int_N=int_N, int_O=int_O)

    # %% 2. Plot Figure 4.10 Heatmaps (Russian labels, >=450 DPI + SVG)
    svg_dir = output_dir / "svg_output"
    svg_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "andrle_fig4_10_heatmap.png"
    out_svg = svg_dir / "andrle_fig4_10_heatmap.svg"
    fig = fit.plot_andrle_heatmap(
        theta_deg, phi_deg, int_N, int_O,
        output_path=out_png,
        close=False,
        lang="ru",
    )
    fig.savefig(out_svg, dpi=450)
    print(f"Saved replicated heatmap figure to: {out_png} and {out_svg}")
    plt.show()


if __name__ == "__main__":
    main()
