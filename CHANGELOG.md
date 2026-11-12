# Changelog

## v1.1.1 - 2026-11-12

- Define two cross-grid references explicitly: a shared sampling-aware native
  reference and an implementation-matched native reference.
- Add implementation-matched sample-space comparisons to the synthetic and
  Penobscot result tables and report their smaller, conservative gains.
- Retain the shared reference as a separate physical-coordinate comparison and
  label both reference definitions in every result row.
- Add numerical aliasing-control ratios and deterministic reference outputs
  under `output/revision_v1.1.1/`.

## v1.1.0 - 2026-11-12

- Add the public Penobscot anti-aliased field-resampling experiment, the
  qualitative time-slice comparison, and the unfiltered aliasing control.
- Add complete cross-grid shape-RMSE tables for THDR, ASAG, TDR, and TAHG.
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
