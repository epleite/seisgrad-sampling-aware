# Revision v1.1.2 reference outputs

This directory contains the deterministic tables and figures used for the
revised manuscript.

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
