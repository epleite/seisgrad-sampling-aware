"""Reproduce the numerical results and figures reported in the manuscript."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    """Run one repository command and stop if it fails."""

    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the synthetic benchmarks and paper figures."
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Regenerate synthetic Figures 1--4 without recomputing stored "
            "benchmark tables; Figure 5 requires the external Penobscot volume."
        ),
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do not run the unit-test suite after reproduction.",
    )
    parser.add_argument(
        "--penobscot-data",
        type=Path,
        help=(
            "Optional path to Penobscot/seismic.npy from Zenodo record "
            "21764042. When omitted, the external-data experiment is skipped."
        ),
    )
    args = parser.parse_args()

    if args.figures_only and args.penobscot_data is not None:
        parser.error(
            "--figures-only cannot be combined with --penobscot-data; "
            "omit --figures-only to regenerate the field experiment and Figure 5"
        )

    if args.figures_only:
        run("scripts/run_channel_benchmark.py", "--figures-only")
        run("scripts/run_benchmark.py", "--figures-only")
        run("scripts/run_revision_experiments.py", "--figures-only")
    else:
        # The fault benchmark uses the channel-margin table when building Fig. 4.
        run("scripts/run_channel_benchmark.py")
        run("scripts/run_benchmark.py")
        run("scripts/run_revision_experiments.py")
        if args.penobscot_data is not None:
            run(
                "scripts/run_penobscot_resampling.py",
                str(args.penobscot_data.resolve()),
            )
        else:
            print(
                "Skipping optional Penobscot experiment; pass --penobscot-data "
                "to reproduce it.",
                flush=True,
            )

    if not args.skip_tests:
        run("-m", "unittest", "discover", "-s", "tests", "-v")


if __name__ == "__main__":
    main()
