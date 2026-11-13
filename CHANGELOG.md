# Changelog

## v1.1.3 - 2026-11-13

- Rename the provisional ``shape RMSE`` metric to the explicit
  ``RMS-normalized error`` while retaining a backward-compatible Python alias.
- Redesign Figures 3--5 with consistent panel labels, redundant visual coding,
  shared legends or color bars, physical axis units, and publication-scale
  typography.

## v1.1.2 - 2026-11-12

- Add a deterministic clean control for the exact 30-degree, 25 x 12.5 m,
  8% channel configuration used by the paired 10 dB noise experiment.
- Clarify that synthetic summary claims use the smaller gain from the shared
  and implementation-matched references separately for each attribute and grid.
- Preserve the v1.1.1 stochastic outputs byte for byte while documenting that
  noise, rather than geometry alone, compresses the clean channel AP gain.

## v1.1.1 - 2026-11-12

- Define two cross-grid references explicitly: a shared sampling-aware native
  reference and an implementation-matched native reference.
- Add implementation-matched sample-space comparisons to the synthetic and
  Penobscot result tables and report both reference definitions.
- Retain the shared reference as a separate physical-coordinate comparison and
  label both reference definitions in every result row.
- Add numerical aliasing-control ratios and deterministic reference outputs
  under `output/revision_v1.1.1/` in that release.

## v1.1.0 - 2026-11-12

- Add the public Penobscot anti-aliased field-resampling experiment, the
  qualitative time-slice comparison, and the unfiltered aliasing control.
- Add complete cross-grid RMS-normalized-error tables for THDR, ASAG, TDR, and TAHG.
- Add 30-realization paired Gaussian and colored/impulsive/footprint tests with
  deterministic seeds and paired bootstrap confidence intervals.
- Add global and spatial velocity-error experiments and report
  `||Gz|| / ||Gh||`.
- Add second- versus fourth-order operator sensitivity with explicit boundary
  treatment and crop sizes.
- Add a checksum-verified range downloader for the external Penobscot amplitude
  array; no third-party data are redistributed in this release.

## v1.0.0 - 2026-08-27

- Initial reproducibility release for the fault and incised-channel benchmarks.
