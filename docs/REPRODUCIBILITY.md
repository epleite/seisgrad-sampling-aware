# Reproducibility guide

## Scientific inputs

The scripts generate two continuous synthetic geological targets and sample
them independently on several seismic grids. The optional field experiment
uses one public external amplitude volume and never uses its labels.

### Fault benchmark

`scripts/run_benchmark.py` builds an 800 m x 800 m x 800 m layered acoustic
model with an oblique normal fault. The depth sampling is 5 m and the post-stack
zero-offset response uses a 30 Hz Ricker wavelet. Fault throws of 25, 50, and
100 m are tested.

The independent sampling cases are:

| Case | dx (m) | dy (m) | dt (s) |
|---|---:|---:|---:|
| reference | 12.5 | 12.5 | 0.002 |
| coarse inline | 25.0 | 12.5 | 0.004 |
| coarse crossline | 12.5 | 25.0 | 0.004 |
| coarse both | 25.0 | 25.0 | 0.004 |

The original benchmark evaluates Gaussian noise at clean, 20, 10, and 5 dB
SNR. `scripts/run_revision_experiments.py` adds 30 paired realizations at 10 dB
for Gaussian white noise and for a documented composite of colored noise,
Student-t impulses, and a periodic acquisition footprint.

### Channel benchmark

`scripts/run_channel_benchmark.py` generates a three-dimensional incised,
meandering channel with a semi-elliptical cross-section and a smoothly varying
impedance contrast. Six planform orientations are tested on the same four
sampling grids. Clean and noisy cases, three impedance contrasts, and an
analytical channel-margin mask provide the qualitative and quantitative tests.

The exact layer properties, geometry equations, wavelet construction, and
sampling parameters are implemented in `src/seisgrad/model.py`,
`src/seisgrad/forward.py`, and the benchmark scripts.

### Public Penobscot field-resampling proxy

The input is `Penobscot/seismic.npy` from the AdaSemSeg processed-data archive
([Zenodo record 21764042](https://doi.org/10.5281/zenodo.21764042)). Its expected
shape and type are `(460, 471, 870)` and `uint8`; its MD5 checksum is
`6afe5db0ffd28cdb0ffae98ee5d1cbdb`. The checksum-verified downloader retrieves
only this 188.5 MB ZIP member by HTTP range request. Labels and the source HDF5
file are not downloaded or used.

The script selects indices `[134:326, 139:331, 243:627]`, yielding the central
`192 x 192 x 384` block. It subtracts the median and divides by the 99th
percentile of absolute amplitude. Native spacings are 12.5 m inline, 25.01 m
crossline, and 4 ms. Before factor-two decimation, the relevant horizontal axis
is filtered by a 31-tap FIR with a Kaiser window (`beta=8.0`) and normalized
cutoff 0.45. Convolution uses reflected boundaries. Two reference definitions
are reported. The shared physical-coordinate comparison uses the sampling-aware
attribute on the filtered native grid for both target implementations. The
conservative implementation-matched comparison uses the native sample-space
attribute for the decimated sample-space result and the native sampling-aware
attribute for the decimated sampling-aware result. Both native references are
sampled at coincident target locations. Comparisons exclude 20 horizontal and
eight temporal target-grid samples at each boundary. A direct, unfiltered
decimation is reported as a separate aliasing control. The reference velocity
is 3000 m/s, with 2550 and 3450 m/s sensitivity cases.

## Attribute inputs and outputs

The public function is:

```python
gradient_attributes(
    amplitude,
    spacing=(dx_m, dy_m, dt_s),
    reference_velocity_m_s=3000.0,
    sampling_aware=True,
    derivative_order=2,
)
```

Input requirements and options:

- `amplitude`: three-dimensional array ordered as `(x, y, time)`;
- `spacing`: positive horizontal distances in metres and time sampling in seconds;
- `reference_velocity_m_s`: positive scalar or array matching `amplitude`;
- `sampling_aware=False`: selects the per-sample comparator;
- `derivative_order=2`: centered three-point main operator;
- `derivative_order=4`: five-point interior sensitivity operator. The two
  samples nearest either boundary retain the order-2 treatment.

The function returns THDR, ASAG, TDR, and TAHG arrays with the same shape as
the input. TDR and TAHG are angles in radians.

## Commands

Run the synthetic workflow and tests from the repository root:

```bash
python scripts/reproduce_paper.py
```

Equivalent explicit sequence:

```bash
python scripts/run_channel_benchmark.py
python scripts/run_benchmark.py
python scripts/run_revision_experiments.py
python -m unittest discover -s tests -v
```

To regenerate synthetic Figures 1--4 from their stored result tables:

```bash
python scripts/reproduce_paper.py --figures-only
```

This fast path leaves Figure 5 unchanged because its panels require the
external Penobscot amplitude volume.

To include the field experiment:

```bash
python -m pip install -e '.[field-data]'
python scripts/fetch_penobscot.py
python scripts/reproduce_paper.py \
  --penobscot-data data/Penobscot/seismic.npy
```

The field data remain outside version control and outside the SeisGrad Zenodo
software archive. `--skip-tests` omits the unit tests. The workflow is CPU-only
and requires no GPU.

## Revision metrics and deterministic protocol

RMS-normalized error independently divides the reference and estimate by their
RMS before computing the RMS difference. It therefore measures spatial-pattern
disagreement rather than a positive global amplitude factor. The deprecated
``shape_rmse`` Python name remains as an alias for backward compatibility.
Pearson correlation is reported alongside it. Paired detection experiments always pass the same noisy amplitude
realization to the sample-space and sampling-aware implementations. Random
seeds are fixed in `scripts/run_revision_experiments.py`; paired percentile
bootstrap intervals use 10,000 resamples.

Cross-grid tables answer two complementary questions. The shared
sampling-aware native reference measures closeness to a single
physical-coordinate target. The implementation-matched native reference
measures stability under resampling within each implementation. Manuscript
synthetic summary claims use the smaller of the two gains separately for each
attribute and grid. Field summary claims use the implementation-matched
comparison, which is the smaller value for all reported field cases.

The non-Gaussian model combines anisotropically colored Gaussian noise
(`sigma=(1.6, 0.8, 0.0)`), clipped Student-t noise with 3 degrees of freedom,
and a sinusoidal footprint with a 12-sample period along the coarse horizontal
axis. Their weights are 0.65, 0.25, and 0.35. The zero-mean composite is
normalized to the requested 10 dB RMS SNR.

## Manuscript figure map

| Manuscript item | Generated file |
|---|---|
| Figure 1 | `output/benchmark_v0.2/submission_figure_1_fault_complete.png` |
| Figure 2 | `output/channel_benchmark_v0.1/channel_manuscript_comparison.png` |
| Revised Figure 3 | `output/revision_v1.1.3/figure_3_rms_normalized_error.png` |
| Revised Figure 4 | `output/revision_v1.1.3/figure_4_paired_ap.png` |
| Figure 5 | `output/revision_v1.1.3/figure_5_penobscot_time_slice.png` |

The manuscript figures are saved as 600-dpi PNG images. CSV and JSON files in
the same directories contain the reference numerical results used to build the
figures and manuscript statistics.

## Expected behavior

A successful full run prints progress for the channel, fault, revision, figure,
field (when requested), and test stages. The eleven unit tests should report
`OK`. Small floating-point differences may occur across platforms or dependency
versions; `requirements.txt` defines the reference environment.
