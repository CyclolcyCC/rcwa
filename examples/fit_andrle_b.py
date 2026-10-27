"""GIXRF Reconstruction & Visualization for Anna Andrle Configuration B (680 eV).

Reconstructs the trapezoid grating with SiO2 oxide and C2O carbon contamination
against the 178-point experimental dataset (N-Kalpha, O-Kalpha, C-Kalpha).
Saves fitting results to a separate file for instant access without re-running.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure xsw is available in any environment
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import xsw.fit as fit

# Set to True to instantly load saved results without re-running the optimization
USE_SAVED = False

# %% 1. Load Experimental Dataset (178 angles, 3 channels: N, O, C)
output_dir = REPO_ROOT / "examples" / "output"
data_file = output_dir / "andrle_fig4_37_experimental_data.npz"
fit_file = output_dir / "fit_andrle_b_results.npz"

full_data = fit.load_andrle_data(data_file)
data = full_data.subsample(step=5)  # 36 angles for fast optimizer convergence

# %% 2. Nominal Starting Parameters (Anna Andrle Thesis Section 4.5.1)
# h=93 nm, w=44.9 nm, swa=82.5 deg, tt=2.47 nm, tg=3.67 nm, tc=2.5 nm
p0 = fit.AndrleBParams(
    h=93.0, w=44.9, swa=82.5, tt=2.47, tg=3.67, tc=2.5,
    sigma_sw=1.5, sigma_vert=1.0,
)

# %% 3. Run Optimization or Load Saved Results
if USE_SAVED and fit_file.exists():
    print(f"Loading pre-calculated fit results from: {fit_file}")
    result = fit.FitResult.load(fit_file)
else:
    print("Running GIXRF reconstruction...")
    result = fit.fit_andrle_structure(data, initial_params=p0, num_slices=10, max_order=12, max_nfev=20)
    result.forward = fit.evaluate_forward(result.params, full_data, num_slices=10, max_order=12)
    result.save(fit_file)
    print(f"Saved fitting results to: {fit_file} and {fit_file.with_suffix('.txt')}")

print(result.summary())

# %% 4. Plot and Save (Single Graph Fig. 4.37 & 4-Panel Summary)
fit.plot_reconstruction_single_graph(full_data, result, output_path=output_dir / "fit_andrle_b_single_graph.png")
fit.plot_reconstruction_summary(full_data, result, num_slices=10, output_path=output_dir / "fit_andrle_b_summary.png")

plt.show()
