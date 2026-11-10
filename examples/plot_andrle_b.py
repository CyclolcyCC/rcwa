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

# %% 3. Plot Single Graph (Fig. 4.37), 4-Panel Summary, and Standalone Reconstructed Profile (RU + SVG)
fig_single = fit.plot_reconstruction_single_graph(
    data, result, output_path=output_dir / "fit_andrle_b_single_graph.png", lang="ru",
)
fig_single.savefig(output_dir / "fit_andrle_b_single_graph.svg")

fig_summary = fit.plot_reconstruction_summary(
    data, result, num_slices=10, output_path=output_dir / "fit_andrle_b_summary.png", lang="ru",
)
fig_summary.savefig(output_dir / "fit_andrle_b_summary.svg")

# Standalone Reconstructed Profile (matching Image 4 in Russian + vector SVG)
struct, _, _, _ = fit.build_andrle_b_sample(result.params, num_slices=10)
fig_prof, ax_prof = plt.subplots(figsize=(5.6, 6.2), layout="constrained")
fit.plot_structure(
    struct,
    ax=ax_prof,
    title=(
        f"Восстановленный профиль\n"
        f"$h={result.params.h:.1f}$ нм, $w={result.params.w:.1f}$ нм, $t_C={result.params.tc:.1f}$ нм"
    ),
    lang="ru",
)
fig_prof.savefig(output_dir / "reconstructed_profile.png", dpi=450)
fig_prof.savefig(output_dir / "reconstructed_profile.svg")
print(f"Saved Russian SVG outputs in: {output_dir}")

plt.show()
