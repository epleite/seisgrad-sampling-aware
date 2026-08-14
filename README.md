# SeisGrad: sampling-aware gradient attributes for 3D seismic volumes

This repository contains the complete Python implementation, synthetic models,
numerical experiments, reference tables, and final figures for the associated
*Computers & Geosciences* Short Note.

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
- `scripts/reproduce_paper.py`: one-command reproduction workflow;
- `tests/`: unit tests;
- `output/`: reference numerical tables and the four 600-dpi manuscript figures;
- `docs/REPRODUCIBILITY.md`: detailed inputs, outputs, options, and figure map.

All inputs are generated synthetically by the repository. No proprietary or
external seismic data are required.

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

After completing the installation, run:

```bash
python -m unittest discover -s tests -v
```

A successful run executes seven tests and ends with `OK`.


## Reproduce the article

Full recomputation, figure generation, and tests:

```bash
python scripts/reproduce_paper.py
```

Fast figure regeneration from the stored result tables:

```bash
python scripts/reproduce_paper.py --figures-only
```

The full workflow runs the channel benchmark first because Figure 4 combines
channel and fault results. Existing files in `output/` are overwritten with
the newly generated results.

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
<emilson@unicamp.br>

## License

MIT. See `LICENSE`.
