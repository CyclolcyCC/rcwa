"""Instant plot viewer for Anna Andrle Configuration B pre-calculated fit results.

Loads saved fitting results without re-running the optimization algorithm.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure xsw is available in any environment
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import xsw.fit as fit

# %% 1. Load Experimental Data & Pre-Calculated Fit Results
output_dir = REPO_ROOT / "examples" / "output"
data_file = output_dir / "andrle_fig4_37_experimental_data.npz"
fit_file = output_dir / "fit_andrle_b_results.npz"

if not fit_file.exists():
    raise FileNotFoundError(
        f"Saved fit results not found at '{fit_file}'. "
        "Please run 'examples/fit_andrle_b.py' once to generate and save the fit results."
    )

data = fit.load_andrle_data(data_file)
result = fit.FitResult.load(fit_file)

# %% 2. Print Saved Reconstruction Summary
print(result.summary())

# %% 3. Plot Single Graph (Fig. 4.37) and 4-Panel Reconstruction Summary
fit.plot_reconstruction_single_graph(data, result)
fit.plot_reconstruction_summary(data, result, num_slices=10)

plt.show()
