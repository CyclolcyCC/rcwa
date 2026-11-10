"""GIXRF vs GEXRF Comparison for Anna Andrle Configuration B.

Compares Grazing Incidence (GIXRF at 680 eV) with Grazing Emission (GEXRF
via the optical reciprocity theorem) for the reconstructed nanograting model.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure xsw is available in any environment
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import xsw
import xsw.fit as fit

# %% 1. Load Reconstructed Model & Angles
output_dir = REPO_ROOT / "examples" / "output"
fit_file = output_dir / "fit_andrle_b_results.npz"
if not fit_file.exists():
    raise FileNotFoundError(f"Fit results not found at {fit_file}. Run examples/fit_andrle_b.py first.")

result = fit.FitResult.load(fit_file)
p = result.params
fwd = result.forward
angles_deg = fwd.angles_deg
angles_rad = np.deg2rad(angles_deg)

# Load experimental data if available for direct visual overlay on GIXRF
data_file = output_dir / "andrle_fig4_37_experimental_data.npz"
data = fit.load_andrle_data(data_file) if data_file.exists() else None

# %% 2. GIXRF Signals (Shared Physical Count Scale from Reconstruction)
gixrf = {"N": fwd.scaled_N, "O": fwd.scaled_O, "C": fwd.scaled_C}
scales = {"N": fwd.scale_N, "O": fwd.scale_O, "C": fwd.scale_C}

# %% 3. GEXRF via Optical Reciprocity (Solving at Characteristic Emission Lines)
# By the Lorentz reciprocity theorem, the exit-angle emission probability is computed
# by solving for the internal field of a time-reversed plane wave incident at exit angle beta
# and emission wavelength lambda_em.
emissions = {
    "N": (392.4, "Si3N4", "#006d2c", "#a1d99b", "#238b45", "o", "N-K$\\alpha$ (392.4 эВ)"),
    "O": (524.9, "SiO2",  "#08519c", "#9ecae1", "#2171b5", "s", "O-K$\\alpha$ (524.9 эВ)"),
    "C": (277.0, "C2O",   "#252525", "#d9d9d9", "#525252", "^", "C-K$\\alpha$ (277.0 эВ)"),
}

gexrf_cache = output_dir / "gixrf_vs_gexrf_data.npz"
if gexrf_cache.exists() and "--recalc" not in sys.argv:
    print(f"Loading cached GEXRF curves from {gexrf_cache.name}...")
    with np.load(gexrf_cache) as cached:
        gexrf = {ch: cached[ch] for ch in emissions}
else:
    print(f"Reconstructed Grating: h={p.h:.1f}nm, w={p.w:.1f}nm, swa={p.swa:.1f}°, tt={p.tt:.2f}nm, tg={p.tg:.2f}nm, tc={p.tc:.2f}nm")
    print("Computing GEXRF angular curves via optical reciprocity...")

    gexrf = {}
    for ch, (E_em, mat_key, _, _, _, _, label) in emissions.items():
        lam_em = 1239.841984 / E_em  # Reciprocal wavelength in nm

        chi_si = xsw.chi_from_formula("Si", density=2.33, energy=E_em)
        chi_n = xsw.chi_from_formula("Si3N4", density=3.1, energy=E_em)
        chi_o = xsw.chi_from_formula("SiO2", density=2.2, energy=E_em)
        chi_c = xsw.chi_from_formula("C2O", density=1.8, energy=E_em)

        struct, mat_n, mat_o, mat_c = fit.build_andrle_b_sample(
            p, num_slices=10, optical_constants=(chi_si, chi_n, chi_o, chi_c)
        )
        target_mat = {"Si3N4": mat_n, "SiO2": mat_o, "C2O": mat_c}[mat_key]
        prep = xsw.prepare(struct, max_order=12)

        raw = np.empty_like(angles_deg)
        for i, beta in enumerate(angles_rad):
            sol = prep.solve(alpha=beta, phi=0.0, wavelength=lam_em, polarization="p")
            raw[i] = sol.fluorescence(target_mat, attenuation=1e-3)

        gexrf[ch] = raw * scales[ch]
        print(f"  Finished GEXRF {label} (peak count: {gexrf[ch].max():.3f})")

    np.savez_compressed(gexrf_cache, **gexrf)

# %% 4. Plot Side-by-Side Comparison: GIXRF vs GEXRF on Shared Physical Count Scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.4), layout="constrained")

exp_map = {
    "O": (data.exp_O, data.sigma_O, "O-K$\\alpha$") if data is not None else None,
    "N": (data.exp_N, data.sigma_N, "N-K$\\alpha$") if data is not None else None,
    "C": (data.exp_C, data.sigma_C, "C-K$\\alpha$") if (data is not None and data.has_carbon) else None,
}

for ch, (_, _, c_fit, c_fill, c_edge, fmt, label) in emissions.items():
    if gixrf[ch] is not None:
        exp_entry = exp_map.get(ch)
        if exp_entry is not None and exp_entry[0] is not None:
            exp_vals, sig_vals, sym = exp_entry
            fit.plot_distinct_fit_and_markers(
                ax1, data.angles_deg, exp_vals, sig_vals, gixrf[ch],
                fmt=fmt, fit_color=c_fit, fill_color=c_fill, edge_color=c_edge,
                exp_label=f"{sym} — эксперимент",
                fit_label=f"{label} — подгонка",
            )
        else:
            ax1.plot(angles_deg, gixrf[ch], "-", color=c_fit, lw=2.2, label=f"{label} — подгонка")

    ax2.plot(angles_deg, gexrf[ch], "-", color=c_fit, lw=2.2, label=label)

# GIXRF Panel (exact title requested)
ax1.set_title("Скользящее падение: эксперимент и подгонка", fontsize=11.5, fontweight="bold")
ax1.set_xlabel(r"Угол скользящего падения $\alpha$ (град)", fontsize=11)
ax1.set_ylabel("Интенсивность флуоресценции (отн. ед.)", fontsize=11)
ax1.set_ylim(-0.02, 1.24)
ax1.legend(loc="upper right", fontsize=8.3, ncol=2, framealpha=0.92)
ax1.grid(True, linestyle=":", alpha=0.5)

# GEXRF Panel (exact title requested)
ax2.set_title("Скользящий выход: симуляция", fontsize=11.5, fontweight="bold")
ax2.set_xlabel(r"Угол скользящего выхода $\beta$ (град)", fontsize=11)
ax2.set_ylabel("Интенсивность флуоресценции (отн. ед.)", fontsize=11)
ax2.set_ylim(-0.02, 1.24)
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
ax2.grid(True, linestyle=":", alpha=0.5)

fig.suptitle(
    f"Конфигурация B (Andrle) — сравнение GI- и GE-геометрий ($p$-поляризация)\n"
    f"Восстановленная решётка ($h={p.h:.1f}$ нм, $w={p.w:.1f}$ нм, $t_C={p.tc:.1f}$ нм)",
    fontsize=12.5,
    fontweight="bold",
)

out_png = output_dir / "gixrf_vs_gexrf_comparison.png"
out_svg = output_dir / "gixrf_vs_gexrf_comparison.svg"
fig.savefig(out_png, dpi=450)
fig.savefig(out_svg)
print(f"\nSaved comparison plot to: {out_png} and {out_svg}")

plt.show()
