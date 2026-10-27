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
    "N": (392.4, "Si3N4", "#006d2c", "#31a354", "N-K$\\alpha$ (392.4 eV)"),
    "O": (524.9, "SiO2",  "#08519c", "#3182bd", "O-K$\\alpha$ (524.9 eV)"),
    "C": (277.0, "C2O",   "#252525", "#737373", "C-K$\\alpha$ (277.0 eV)"),
}

print(f"Reconstructed Grating: h={p.h:.1f}nm, w={p.w:.1f}nm, swa={p.swa:.1f}°, tt={p.tt:.2f}nm, tg={p.tg:.2f}nm, tc={p.tc:.2f}nm")
print("Computing GEXRF angular curves via optical reciprocity...")

gexrf = {}
for ch, (E_em, mat_key, _, _, label) in emissions.items():
    lam_em = 1239.841984 / E_em  # Reciprocal wavelength in nm

    # Optical constants at the specific emission line energy
    chi_si = xsw.chi_from_formula("Si", density=2.33, energy=E_em)
    chi_n = xsw.chi_from_formula("Si3N4", density=3.1, energy=E_em)
    chi_o = xsw.chi_from_formula("SiO2", density=2.2, energy=E_em)
    chi_c = xsw.chi_from_formula("C2O", density=1.8, energy=E_em)

    # Build multislice sample with optical constants at emission energy
    struct, mat_n, mat_o, mat_c = fit.build_andrle_b_sample(
        p, num_slices=10, optical_constants=(chi_si, chi_n, chi_o, chi_c)
    )
    mat_map = {"Si3N4": mat_n, "SiO2": mat_o, "C2O": mat_c}
    target_mat = mat_map[mat_key]

    prep = xsw.prepare(struct, max_order=12)

    # In GEXRF, attenuation is the penetration attenuation of the incident excitation beam (680 eV)
    raw = np.empty_like(angles_deg)
    for i, beta in enumerate(angles_rad):
        sol = prep.solve(alpha=beta, phi=0.0, wavelength=lam_em, polarization="p")
        raw[i] = sol.fluorescence(target_mat, attenuation=1e-3)

    # Scale using the physical elemental factor c_k to preserve absolute count proportions
    gexrf[ch] = raw * scales[ch]
    print(f"  Finished GEXRF {label} (peak count: {gexrf[ch].max():.3f})")

# %% 4. Plot Side-by-Side Comparison: GIXRF vs GEXRF on Shared Physical Count Scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")

# Experimental data overlay on GIXRF (matching Fig. 4.37 / fit_andrle_b_single_graph)
if data is not None:
    ax1.errorbar(data.angles_deg, data.exp_O, yerr=data.sigma_O, fmt="s", color="#3182bd",
                  markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.45, label="O-K$\\alpha$ Exp")
    ax1.errorbar(data.angles_deg, data.exp_N, yerr=data.sigma_N, fmt="o", color="#31a354",
                  markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.45, label="N-K$\\alpha$ Exp")
    if data.has_carbon:
        ax1.errorbar(data.angles_deg, data.exp_C, yerr=data.sigma_C, fmt="^", color="#737373",
                      markersize=2.2, capsize=0, elinewidth=0.6, alpha=0.45, label="C-K$\\alpha$ Exp")

for ch, (_, _, color, _, label) in emissions.items():
    if gixrf[ch] is not None:
        ax1.plot(angles_deg, gixrf[ch], "-", color=color, lw=2.2, label=f"{label} Fit")
    ax2.plot(angles_deg, gexrf[ch], "-", color=color, lw=2.2, label=label)

# GIXRF Panel
ax1.set_title("GIXRF: Grazing Incidence ($E_{\\mathrm{in}} = 680$ eV, $\\lambda = 1.82$ nm)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Grazing angle of incidence $\\alpha$ (deg)", fontsize=11)
ax1.set_ylabel("Fluorescence intensity (counts)", fontsize=11)
ax1.set_ylim(-0.02, 1.18)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax1.grid(True, linestyle=":", alpha=0.5)

# GEXRF Panel
ax2.set_title("GEXRF: Grazing Emission (Optical Reciprocity at $\\lambda_{\\mathrm{em}}$)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Grazing angle of emission $\\beta$ (deg)", fontsize=11)
ax2.set_ylabel("Fluorescence intensity (counts)", fontsize=11)
ax2.set_ylim(-0.02, 1.18)
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax2.grid(True, linestyle=":", alpha=0.5)

fig.suptitle(
    f"Anna Andrle Configuration B — GIXRF vs GEXRF Comparison (p-pol)\n"
    f"Reconstructed Grating ($h={p.h:.1f}$ nm, $w={p.w:.1f}$ nm, $t_C={p.tc:.1f}$ nm)",
    fontsize=12,
    fontweight="bold",
)

out_file = output_dir / "gixrf_vs_gexrf_comparison.png"
fig.savefig(out_file, dpi=180)
print(f"\nSaved comparison plot to: {out_file}")

plt.show()
