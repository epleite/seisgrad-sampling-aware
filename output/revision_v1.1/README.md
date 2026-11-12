# Revision v1.1 reference outputs

These files are the checked reference outputs for the ACAGS-D-26-00290
revision. All stochastic tables retain realization-level values and fixed seeds
so that summary statistics can be audited independently.

| File | Contents |
|---|---|
| `cross_grid_shape_rmse.csv` | THDR, ASAG, TDR, and TAHG on all three target grids for both implementations |
| `velocity_sensitivity.csv` | global and spatial velocity perturbations plus `||Gz|| / ||Gh||` |
| `operator_sensitivity.csv` | order 2 versus order 4 at horizontal crops of 2, 4, 6, and 8 samples |
| `fault_paired_realizations.csv` | 30 paired values for every fault throw, noise model, and attribute |
| `channel_paired_realizations.csv` | 30 paired channel-margin THDR values per noise model |
| `paired_bootstrap_summary.csv` | paired means and 95% percentile-bootstrap intervals |
| `penobscot_resampling.csv` | anti-aliased field results, +/-15% velocity cases, and unfiltered aliasing controls |

The Penobscot table reproduces the manuscript values at 3000 m/s. For the
crossline-decimated 12.5 m x 50.02 m grid, sample-space versus sampling-aware
shape RMSE is 0.38659 versus 0.12117 (THDR), 0.25445 versus 0.00959 (ASAG),
0.26547 versus 0.04024 (TDR), and 0.74475 versus 0.35555 (TAHG).

The current deterministic paired-noise protocol gives THDR AP changes of
-0.0030, +0.0066, and +0.0285 for 25, 50, and 100 m throws under Gaussian
noise, and -0.0036, +0.0058, and +0.0283 under the colored/impulsive/footprint
model. In the channel experiment the corresponding means (95% CI) are +0.0066
(-0.0008, +0.0142) and +0.0130 (+0.0070, +0.0192).
