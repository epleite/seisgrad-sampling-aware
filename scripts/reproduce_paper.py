"""Reproduce the numerical results and figures reported in the Short Note."""

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
        help="Regenerate the paper figures without recomputing stored benchmark tables.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do not run the unit-test suite after reproduction.",
    )
    args = parser.parse_args()

    if args.figures_only:
        run("scripts/run_channel_benchmark.py", "--figures-only")
        run("scripts/run_benchmark.py", "--figures-only")
    else:
        # The fault benchmark uses the channel-margin table when building Fig. 4.
        run("scripts/run_channel_benchmark.py")
        run("scripts/run_benchmark.py")

    if not args.skip_tests:
        run("-m", "unittest", "discover", "-s", "tests", "-v")


if __name__ == "__main__":
    main()
