# Reference outputs

This directory contains the numerical tables and four high-resolution figures
reported in the manuscript. They are retained so that users can inspect the
reference results and run the fast `--figures-only` workflow without first
recomputing every benchmark.

Running `python scripts/reproduce_paper.py` overwrites these files with a fresh
reproduction. The benchmark scripts may also create additional diagnostic
figures and a channel report that are not required by the manuscript.
