# Revision v1.1.3 reference outputs

This directory contains the deterministic tables and figures used for the
revised manuscript.

The scale-invariant comparison is reported as ``RMS-normalized error``. The
reference and estimate are independently divided by their RMS before the RMS
difference is calculated.

The cross-grid CSV files label every row with one of two reference definitions:

- `shared_sampling_aware_native`: both target implementations are compared with
  the sampling-aware result on the native/reference grid. This asks how closely
  each target reproduces one physical-coordinate attribute.
- `implementation_matched_native`: sample-space and sampling-aware target
  results are each compared with the same implementation on the native grid.
  This isolates stability under resampling within an implementation. The
  revised manuscript reports both definitions and uses the smaller percentage
  gain separately for each attribute and grid in synthetic summary claims.

`penobscot_resampling.csv` also labels anti-aliased and unfiltered decimation,
so unit scaling and aliasing effects can be inspected separately.

`channel_clean_control.csv` evaluates the same 30-degree channel, 25 x 12.5 m
grid, and 8% contrast used by the paired 10 dB experiment without added noise.

## Included files

- `cross_grid_rms_normalized_error.csv`: synthetic cross-grid comparisons for
  all four attributes and both reference definitions.
- `velocity_sensitivity.csv`: global and laterally varying velocity tests.
- `operator_sensitivity.csv`: second- and fourth-order derivative tests.
- `fault_paired_realizations.csv`, `channel_paired_realizations.csv`, and
  `paired_bootstrap_summary.csv`: paired 30-realization results and confidence
  intervals.
- `channel_clean_control.csv`: exact noiseless control for the noisy channel
  configuration.
- `penobscot_resampling.csv`: filtered and unfiltered field-resampling results.
- `figure_3_rms_normalized_error.png`: implementation-matched cross-grid error.
- `figure_4_paired_ap.png`: paired fault-detection changes with 95% confidence
  intervals.
- `figure_5_penobscot_time_slice.png`: native-reference, conventional, and
  sampling-aware Penobscot time slices for THDR and TDR. The scientific image
  panels are unchanged from the validated field run; only their layout,
  labeling, physical axes, and shared color bars were revised for the
  manuscript. Exact recomputation requires the external Penobscot amplitude
  file documented in `docs/REPRODUCIBILITY.md`.
