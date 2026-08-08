# Exploratory incised-channel benchmark

## Purpose

This experiment tests the numerical and interpretive behavior of sampling-aware gradient attributes on a planform channel target.

## Design

- Continuous incised meandering channel with semi-elliptical cross-section.
- Smoothly varying impedance contrast along the channel.
- Four independently generated grids and six channel orientations.
- Channel-margin ground truth evaluated with average precision and best F1.
- Low, medium, and high impedance contrasts, with clean and 10 dB cases.
- Input amplitude and unsteered semblance-discontinuity baselines.

## Cross-grid invariance

| Attribute | Sample-space NRMSE | Sampling-aware NRMSE | Reduction |
|---|---:|---:|---:|
| THDR | 0.602 | 0.514 | 14.6% |
| ASAG | 0.692 | 0.168 | 75.8% |
| TDR | 0.123 | 0.087 | 29.2% |
| TAHG | 0.996 | 0.880 | 11.6% |

## Channel-margin recovery on anisotropic grids

Values are means over six orientations and the two grids with interchanged 25 m and 12.5 m horizontal sampling. Standard deviations quantify orientation sensitivity.

| Attribute | Sample-space AP | Sampling-aware AP | Sample-space SD | Sampling-aware SD |
|---|---:|---:|---:|---:|
| THDR | 0.792 | 0.872 | 0.049 | 0.035 |
| ASAG | 0.429 | 0.416 | 0.056 | 0.024 |
| TDR | 0.065 | 0.065 | 0.007 | 0.007 |
| TAHG | 0.602 | 0.617 | 0.090 | 0.105 |

## Medium-contrast clean reference case

| Score | Average precision |
|---|---:|
| baseline:input_amplitude | 0.384 |
| baseline:semblance_discontinuity | 0.837 |
| sample_space:asag | 0.486 |
| sample_space:tahg_magnitude | 0.518 |
| sample_space:tdr_magnitude | 0.061 |
| sample_space:tdr_zero_conditioned | 0.907 |
| sample_space:thdr | 0.916 |
| sampling_aware:asag | 0.473 |
| sampling_aware:tahg_magnitude | 0.517 |
| sampling_aware:tdr_magnitude | 0.061 |
| sampling_aware:tdr_zero_conditioned | 0.909 |
| sampling_aware:thdr | 0.916 |

## Noise stress test

Average precision on the reference grid. The sampling-aware and sample-space THDR rankings coincide on this horizontally isotropic grid, so one THDR value is shown.

| Contrast | Condition | Input amplitude | THDR | Semblance discontinuity |
|---:|---|---:|---:|---:|
| 3% | clean | 0.386 | 0.916 | 0.866 |
| 3% | 10 dB | 0.205 | 0.298 | 0.657 |
| 6% | clean | 0.384 | 0.916 | 0.837 |
| 6% | 10 dB | 0.297 | 0.627 | 0.729 |
| 12% | clean | 0.378 | 0.916 | 0.741 |
| 12% | 10 dB | 0.353 | 0.875 | 0.674 |

## Scope of interpretation

- Numerical invariance criterion met: **yes**.
- THDR margin-recovery/orientation criterion met: **yes**.
- TDR zero-contour conditioning is reported separately and must not be confused with the raw TDR magnitude.
- Physical scaling can reduce grid bias but cannot remove band limitation, tuning, or discretization error.
- At 10 dB SNR, semblance outperforms THDR for the 3% and 6% contrasts; the channel result therefore supports complementary use, not universal superiority.

## Benchmark conclusion

The channel benchmark supports the sampling/azimuth test and gives THDR a clear interpretive role. It does not make all four raw attributes equally useful: ASAG is stable but not margin-specific, raw TDR magnitude is not an edge score, and TAHG remains oscillatory. The channel and fault experiments are therefore complementary demonstrations rather than evidence that one attribute is universally superior.
