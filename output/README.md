# Reference outputs

This directory contains the numerical tables and high-resolution figures
reported in the manuscript. They are retained so that users can inspect the
reference results and run the fast `--figures-only` workflow without first
recomputing every benchmark.

Running `python scripts/reproduce_paper.py` overwrites these files with a fresh
reproduction. The benchmark scripts may also create additional diagnostic
figures and a channel report that are not required by the manuscript.

`revision_v1.1.2/` contains the complete cross-grid shape-RMSE table, velocity
and operator sensitivity, all paired realization-level values, bootstrap
summaries, the exact clean-channel control, the Penobscot resampling and
aliasing-control table, and Figures 3--5 introduced or updated during revision.
Every cross-grid row labels whether it uses the shared sampling-aware native
reference or an implementation-matched native reference.
