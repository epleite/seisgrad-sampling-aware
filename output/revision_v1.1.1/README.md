# Revision v1.1.1 reference outputs

This directory contains the deterministic tables and figures used for the
revised manuscript.

The cross-grid CSV files label every row with one of two reference definitions:

- `shared_sampling_aware_native`: both target implementations are compared with
  the sampling-aware result on the native/reference grid. This asks how closely
  each target reproduces one physical-coordinate attribute.
- `implementation_matched_native`: sample-space and sampling-aware target
  results are each compared with the same implementation on the native grid.
  This isolates stability under resampling within an implementation. The
  revised manuscript uses these smaller, conservative percentage gains as its
  primary numerical claims.

`penobscot_resampling.csv` also labels anti-aliased and unfiltered decimation,
so unit scaling and aliasing effects can be inspected separately.
