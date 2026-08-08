# Reproducibility guide

## Scientific inputs

No external dataset is used. The scripts generate two continuous synthetic
geological targets and sample them independently on several seismic grids.

### Fault benchmark

`scripts/run_benchmark.py` builds an 800 m × 800 m × 800 m layered acoustic
model with an oblique normal fault. The default depth sampling is 5 m and the
post-stack zero-offset response uses a 30 Hz Ricker wavelet. Fault throws of
25, 50, and 100 m are tested. Gaussian noise is evaluated at clean, 20, 10,
and 5 dB SNR, with five realizations for each noisy condition.

The independent sampling cases are:

| Case | dx (m) | dy (m) | dt (s) |
|---|---:|---:|---:|
| reference | 12.5 | 12.5 | 0.002 |
| coarse inline | 25.0 | 12.5 | 0.004 |
| coarse crossline | 12.5 | 25.0 | 0.004 |
| coarse both | 25.0 | 25.0 | 0.004 |

Using a nominal velocity of 3000 m/s, these settings give the voxel aspect
ratios discussed in the manuscript. Random seeds are fixed in the script.

### Channel benchmark

`scripts/run_channel_benchmark.py` generates a three-dimensional incised,
meandering channel with a semi-elliptical cross-section and a smoothly varying
impedance contrast. Six planform orientations are tested on the same four
sampling grids. Clean and noisy cases, three impedance contrasts, and an
analytical channel-margin mask provide the qualitative and quantitative tests.

The exact layer properties, geometry equations, wavelet construction, and
sampling parameters are implemented in `src/seisgrad/model.py`,
`src/seisgrad/forward.py`, and the two benchmark scripts.

## Attribute inputs and outputs

The public function is:

```python
gradient_attributes(
    amplitude,
    spacing=(dx_m, dy_m, dt_s),
    reference_velocity_m_s=3000.0,
    sampling_aware=True,
)
```

Input requirements:

- `amplitude`: a three-dimensional array ordered as `(x, y, time)`;
- `spacing`: positive horizontal distances in metres and time sampling in seconds;
- `reference_velocity_m_s`: a positive scalar or an array matching `amplitude`;
- `sampling_aware=False`: selects the per-sample comparator.

The function returns arrays for THDR, ASAG, TDR, and TAHG. All outputs have the
same shape as `amplitude`; TDR and TAHG are angles in radians.

## Commands

Run everything from the repository root:

```bash
python scripts/reproduce_paper.py
```

Equivalent explicit sequence:

```bash
python scripts/run_channel_benchmark.py
python scripts/run_benchmark.py
python -m unittest discover -s tests -v
```

To regenerate only the four manuscript figures from the stored result tables:

```bash
python scripts/reproduce_paper.py --figures-only
```

Use `--skip-tests` with the orchestration script only when test execution is
not desired. The workflow is CPU-only and requires no GPU. Runtime and memory
depend on the platform; the full workflow holds multiple 3D volumes in memory,
whereas figure-only regeneration is substantially lighter.

## Manuscript figure map

| Manuscript item | Generated file |
|---|---|
| Figure 1 | `output/benchmark_v0.2/submission_figure_1_fault_complete.png` |
| Figure 2 | `output/benchmark_v0.2/benchmark_figure_2_invariance.png` |
| Figure 3 | `output/channel_benchmark_v0.1/channel_manuscript_comparison.png` |
| Figure 4 | `output/benchmark_v0.2/submission_figure_4_joint_metrics.png` |

The four files are saved as 600-dpi PNG images by the final plotting routines.
The CSV and JSON files in the same directories are the reference numerical
results used to build the figures and manuscript summary statistics.

## Expected behavior

A successful run prints progress for the channel benchmark, fault benchmark,
figure generation, and tests. The seven unit tests should report `OK`. Small
floating-point differences may occur across platforms or dependency versions;
the pinned versions in `requirements.txt` define the reference environment.
