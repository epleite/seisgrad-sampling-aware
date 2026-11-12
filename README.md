# SeisGrad: sampling-aware gradient attributes for 3D seismic volumes

This repository contains the complete Python implementation, synthetic models,
numerical experiments, reference tables, and final figures for the associated
*Applied Computing and Geosciences* manuscript, "Grid dependence of
gradient-derived seismic attributes: a sampling-aware formulation."

The study tests whether THDR, ASAG, TDR, and TAHG change when the same synthetic
geology is represented on seismic grids with different inline, crossline, and
time sampling intervals. The sampling-aware implementation differentiates in
physical coordinates and converts the time derivative using a reference
velocity. The conventional comparator differentiates per sample.

## Contents

- `src/seisgrad/`: attributes, synthetic forward modelling, geological models,
  baselines, and metrics;
- `scripts/run_channel_benchmark.py`: incised-channel benchmark and Figure 3;
- `scripts/run_benchmark.py`: fault benchmark and Figures 1, 2, and 4;
- `scripts/run_revision_experiments.py`: cross-grid shape RMSE, paired
  30-realization noise tests, velocity sensitivity, and operator sensitivity;
- `scripts/fetch_penobscot.py`: checksum-verified retrieval of only the public
  amplitude array needed for the field experiment;
- `scripts/run_penobscot_resampling.py`: anti-aliased Penobscot resampling,
  velocity sensitivity, aliasing control, and Figure 5;
- `scripts/reproduce_paper.py`: one-command reproduction workflow;
- `tests/`: unit tests;
- `output/`: reference numerical tables and the manuscript figures;
- `docs/REPRODUCIBILITY.md`: detailed inputs, outputs, options, and figure map.

The fault and channel inputs are generated synthetically. The optional field
experiment uses the public AdaSemSeg Penobscot amplitude volume
([DOI 10.5281/zenodo.21764042](https://doi.org/10.5281/zenodo.21764042)); the
data are not redistributed here and labels are not used.

## Installation

Python 3.11 or newer is required. The reference environment used Python 3.12.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Quick test

After installation, run:

```bash
python -m unittest discover -s tests -v
```

A successful run executes eleven tests and ends with `OK`.

## Reproduce the article

Full recomputation, figure generation, and tests:

```bash
python scripts/reproduce_paper.py
```

Fast figure regeneration from the stored result tables:

```bash
python scripts/reproduce_paper.py --figures-only
```

The command also runs the revision experiments. The full workflow runs the
channel benchmark first because the joint benchmark figure combines channel
and fault results. Existing files in `output/` are overwritten.

### Optional public Penobscot experiment

Fetch only the 188.5 MB amplitude member from the 1.1 GB public archive by
HTTP range request, verify its checksum, and run the field experiment:

```bash
python -m pip install -e '.[field-data]'
python scripts/fetch_penobscot.py
python scripts/reproduce_paper.py \
  --penobscot-data data/Penobscot/seismic.npy
```

The experiment selects the central `192 x 192 x 384` subvolume, median-centres
it, scales it by its 99th absolute percentile, filters the axis to be decimated
with a 31-tap Kaiser FIR (`beta=8.0`, normalized cutoff `0.45`), and excludes
20 horizontal and eight temporal target-grid samples at the boundaries.

## Minimal API example

```python
from seisgrad.attributes import gradient_attributes

attributes = gradient_attributes(
    amplitude_volume,
    spacing=(dx_metres, dy_metres, dt_seconds),
    reference_velocity_m_s=velocity_volume_or_scalar,
    sampling_aware=True,
)
```

The returned dictionary contains `thdr`, `asag`, `tdr`, and `tahg` arrays with
the same shape as the input volume. The required volume order is `(x, y, time)`.

## Author

Emilson Pereira Leite  
Department of Geology and Natural Resources, Institute of Geosciences,
University of Campinas, Brazil  

## License

MIT. See `LICENSE`.

## Citation and archive

The DOI for all SeisGrad versions is
[10.5281/zenodo.22127203](https://doi.org/10.5281/zenodo.22127203). See
`CITATION.cff` for machine-readable citation metadata.
