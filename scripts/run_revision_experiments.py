"""Reproduce the synthetic experiments added for ACAGS-D-26-00290 revision.

The script writes scale-invariant cross-grid errors, velocity and operator
sensitivity, a deterministic clean-channel control, paired 30-realization fault
and channel tests, bootstrap confidence intervals, and the two summary figures
introduced in the revision.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_benchmark as fault
import run_channel_benchmark as channel
from seisgrad.attributes import _components, gradient_attributes
from seisgrad.baselines import unsteered_semblance_discontinuity
from seisgrad.metrics import (
    add_colored_impulsive_noise,
    add_gaussian_noise,
    average_precision,
    correlation,
    paired_bootstrap_interval,
    shape_rmse,
)


OUTPUT = ROOT / "output" / "revision_v1.1.2"
IMPLEMENTATIONS = (("sample_space", False), ("sampling_aware", True))
ATTRIBUTES = ("thdr", "asag", "tdr", "tahg")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def score(name: str, values: np.ndarray) -> np.ndarray:
    return np.abs(values) if name in {"tdr", "tahg"} else values


def cross_grid_shape_experiment() -> list[dict[str, object]]:
    reference_case = fault.SAMPLING_CASES[0]
    _, reference_seismic, _, reference_velocity, _ = fault.generate_case(
        reference_case, throw_m=50.0
    )
    reference_attributes = {
        mode: gradient_attributes(
            reference_seismic,
            spacing=(
                reference_case.dx_m,
                reference_case.dy_m,
                reference_case.dt_s,
            ),
            reference_velocity_m_s=reference_velocity,
            sampling_aware=aware,
        )
        for mode, aware in IMPLEMENTATIONS
    }

    rows: list[dict[str, object]] = []
    for case in fault.SAMPLING_CASES[1:]:
        _, seismic, _, velocity, _ = fault.generate_case(case, throw_m=50.0)
        factors = (
            int(round(case.dx_m / reference_case.dx_m)),
            int(round(case.dy_m / reference_case.dy_m)),
            int(round(case.dt_s / reference_case.dt_s)),
        )
        for mode, aware in IMPLEMENTATIONS:
            attributes = gradient_attributes(
                seismic,
                spacing=(case.dx_m, case.dy_m, case.dt_s),
                reference_velocity_m_s=velocity,
                sampling_aware=aware,
            )
            for name in ATTRIBUTES:
                reference_modes = (
                    ("shared_sampling_aware_native", "sampling_aware"),
                    ("implementation_matched_native", mode),
                )
                for reference_definition, reference_mode in reference_modes:
                    reference = reference_attributes[reference_mode][name][
                        :: factors[0], :: factors[1], :: factors[2]
                    ]
                    reference_crop, estimate_crop = fault.common_interior(
                        reference, attributes[name]
                    )
                    rows.append(
                        {
                            "sampling_case": case.name,
                            "dx_m": case.dx_m,
                            "dy_m": case.dy_m,
                            "dt_s": case.dt_s,
                            "reference_definition": reference_definition,
                            "implementation": mode,
                            "attribute": name,
                            "shape_rmse": shape_rmse(
                                reference_crop, estimate_crop
                            ),
                            "correlation": correlation(
                                reference_crop, estimate_crop
                            ),
                        }
                    )
    write_csv(OUTPUT / "cross_grid_shape_rmse.csv", rows)
    return rows


def velocity_sensitivity_experiment() -> list[dict[str, object]]:
    """Evaluate target-grid velocity bias against a known-field reference."""

    reference_case, target_case = fault.SAMPLING_CASES[:2]
    _, reference_seismic, _, reference_velocity, _ = fault.generate_case(
        reference_case, throw_m=50.0
    )
    _, target_seismic, _, target_velocity, _ = fault.generate_case(
        target_case, throw_m=50.0
    )
    factors = (2, 1, 2)
    margin = (3, 3, 8)
    reference = gradient_attributes(
        reference_seismic,
        spacing=(
            reference_case.dx_m,
            reference_case.dy_m,
            reference_case.dt_s,
        ),
        reference_velocity_m_s=reference_velocity,
        sampling_aware=True,
    )["asag"][:: factors[0], :: factors[1], :: factors[2]]
    target_known = gradient_attributes(
        target_seismic,
        spacing=(target_case.dx_m, target_case.dy_m, target_case.dt_s),
        reference_velocity_m_s=target_velocity,
        sampling_aware=True,
    )["asag"]

    ny = target_velocity.shape[1]
    smooth = 1.0 + 0.15 * np.sin(2.0 * np.pi * np.arange(ny) / ny)
    nx = target_velocity.shape[0]
    step = 1.0 + np.where(np.arange(nx) < nx / 2.0, -0.15, 0.15)
    cases: tuple[tuple[str, np.ndarray], ...] = (
        ("known_field", target_velocity),
        ("global_minus_10pct", target_velocity * 0.90),
        ("global_plus_10pct", target_velocity * 1.10),
        ("global_minus_15pct", target_velocity * 0.85),
        ("global_plus_15pct", target_velocity * 1.15),
        ("smooth_lateral_plusminus_15pct", target_velocity * smooth[None, :, None]),
        ("sharp_lateral_step_plusminus_15pct", target_velocity * step[:, None, None]),
    )

    rows: list[dict[str, object]] = []
    for name, velocity in cases:
        estimate = gradient_attributes(
            target_seismic,
            spacing=(target_case.dx_m, target_case.dy_m, target_case.dt_s),
            reference_velocity_m_s=velocity,
            sampling_aware=True,
        )["asag"]
        ref_crop, estimate_crop = fault.common_interior(reference, estimate, margin)
        known_crop, velocity_crop = fault.common_interior(
            target_known, estimate, margin
        )
        rows.append(
            {
                "velocity_case": name,
                "cross_grid_shape_rmse": shape_rmse(ref_crop, estimate_crop),
                "cross_grid_correlation": correlation(ref_crop, estimate_crop),
                "velocity_only_shape_rmse": shape_rmse(known_crop, velocity_crop),
            }
        )

    components = _components(
        target_seismic,
        spacing=(target_case.dx_m, target_case.dy_m, target_case.dt_s),
        reference_velocity_m_s=target_velocity,
        sampling_aware=True,
        derivative_order=2,
    )
    slices = (slice(3, -3), slice(3, -3), slice(8, -8))
    gh = np.hypot(components.gx[slices], components.gy[slices])
    ratio = np.linalg.norm(components.gz[slices].ravel()) / np.linalg.norm(gh.ravel())
    for row in rows:
        row["target_l2_gz_over_gh"] = float(ratio)
    write_csv(OUTPUT / "velocity_sensitivity.csv", rows)
    return rows


def operator_sensitivity_experiment() -> list[dict[str, object]]:
    reference_case, target_case = fault.SAMPLING_CASES[:2]
    _, reference_seismic, _, reference_velocity, _ = fault.generate_case(
        reference_case, throw_m=50.0
    )
    _, target_seismic, _, target_velocity, _ = fault.generate_case(
        target_case, throw_m=50.0
    )
    rows: list[dict[str, object]] = []
    for order in (2, 4):
        reference_attributes = gradient_attributes(
            reference_seismic,
            spacing=(
                reference_case.dx_m,
                reference_case.dy_m,
                reference_case.dt_s,
            ),
            reference_velocity_m_s=reference_velocity,
            sampling_aware=True,
            derivative_order=order,
        )
        target_attributes = gradient_attributes(
            target_seismic,
            spacing=(target_case.dx_m, target_case.dy_m, target_case.dt_s),
            reference_velocity_m_s=target_velocity,
            sampling_aware=True,
            derivative_order=order,
        )
        for horizontal_crop in (2, 4, 6, 8):
            margin = (horizontal_crop, horizontal_crop, 8)
            for name in ATTRIBUTES:
                reference, estimate = fault.common_interior(
                    reference_attributes[name][::2, :, ::2],
                    target_attributes[name],
                    margin,
                )
                rows.append(
                    {
                        "derivative_order": order,
                        "horizontal_crop_samples": horizontal_crop,
                        "temporal_crop_samples": 8,
                        "attribute": name,
                        "shape_rmse": shape_rmse(reference, estimate),
                        "correlation": correlation(reference, estimate),
                    }
                )
    write_csv(OUTPUT / "operator_sensitivity.csv", rows)
    return rows


def _paired_fault_rows() -> list[dict[str, object]]:
    case = fault.SAMPLING_CASES[1]
    evaluation = (slice(4, -4), slice(4, -4), slice(8, -8))
    rows: list[dict[str, object]] = []
    for throw_m in (25.0, 50.0, 100.0):
        _, clean, _, velocity, fault_mask = fault.generate_case(
            case, throw_m=throw_m
        )
        labels = fault_mask[evaluation]
        for noise_name, noise_function, seed_offset in (
            ("gaussian", add_gaussian_noise, 100),
            ("colored_impulsive_footprint", add_colored_impulsive_noise, 200),
        ):
            for realization in range(30):
                seed = (
                    20260807
                    + int(throw_m) * 1000
                    + seed_offset
                    + realization
                )
                noisy = noise_function(
                    clean, 10.0, rng=np.random.default_rng(seed)
                )
                paired: dict[str, dict[str, float]] = {}
                for implementation, aware in IMPLEMENTATIONS:
                    attributes = gradient_attributes(
                        noisy,
                        spacing=(case.dx_m, case.dy_m, case.dt_s),
                        reference_velocity_m_s=velocity,
                        sampling_aware=aware,
                    )
                    paired[implementation] = {
                        name: average_precision(
                            labels, score(name, attributes[name][evaluation])
                        )
                        for name in ATTRIBUTES
                    }
                for name in ATTRIBUTES:
                    conventional = paired["sample_space"][name]
                    aware = paired["sampling_aware"][name]
                    rows.append(
                        {
                            "throw_m": throw_m,
                            "noise_model": noise_name,
                            "snr_db": 10.0,
                            "realization": realization,
                            "attribute": name,
                            "sample_space_ap": conventional,
                            "sampling_aware_ap": aware,
                            "paired_ap_difference": aware - conventional,
                        }
                    )
    write_csv(OUTPUT / "fault_paired_realizations.csv", rows)
    return rows


def _paired_channel_rows() -> list[dict[str, object]]:
    case = channel.SAMPLING_CASES[1]
    model, clean, _, velocity, top = channel.generate_case(
        case,
        orientation_deg=30.0,
        contrast_fraction=channel.REFERENCE_CONTRAST,
    )
    labels, evaluation = channel.evaluation_masks(model)
    rows: list[dict[str, object]] = []
    for noise_name, noise_function, seed_offset in (
        ("gaussian", add_gaussian_noise, 100),
        ("colored_impulsive_footprint", add_colored_impulsive_noise, 200),
    ):
        for realization in range(30):
            seed = 20260808 + seed_offset + realization
            noisy = noise_function(clean, 10.0, rng=np.random.default_rng(seed))
            semblance = unsteered_semblance_discontinuity(noisy)
            paired: dict[str, float] = {}
            for implementation, aware in IMPLEMENTATIONS:
                attributes = gradient_attributes(
                    noisy,
                    spacing=(case.dx_m, case.dy_m, case.dt_s),
                    reference_velocity_m_s=velocity,
                    sampling_aware=aware,
                )
                maps = channel.score_maps(
                    noisy, attributes, semblance, top, case.dt_s
                )
                paired[implementation] = average_precision(
                    labels[evaluation], maps["thdr"][evaluation]
                )
            rows.append(
                {
                    "noise_model": noise_name,
                    "snr_db": 10.0,
                    "realization": realization,
                    "attribute": "thdr",
                    "sample_space_ap": paired["sample_space"],
                    "sampling_aware_ap": paired["sampling_aware"],
                    "paired_ap_difference": (
                        paired["sampling_aware"] - paired["sample_space"]
                    ),
                }
            )
    write_csv(OUTPUT / "channel_paired_realizations.csv", rows)
    return rows


def clean_channel_control() -> list[dict[str, object]]:
    """Evaluate the exact channel configuration used by the noisy paired test."""

    case = channel.SAMPLING_CASES[1]
    model, clean, _, velocity, top = channel.generate_case(
        case,
        orientation_deg=30.0,
        contrast_fraction=channel.REFERENCE_CONTRAST,
    )
    labels, evaluation = channel.evaluation_masks(model)
    semblance = unsteered_semblance_discontinuity(clean)
    paired: dict[str, float] = {}
    for implementation, aware in IMPLEMENTATIONS:
        attributes = gradient_attributes(
            clean,
            spacing=(case.dx_m, case.dy_m, case.dt_s),
            reference_velocity_m_s=velocity,
            sampling_aware=aware,
        )
        maps = channel.score_maps(clean, attributes, semblance, top, case.dt_s)
        paired[implementation] = average_precision(
            labels[evaluation], maps["thdr"][evaluation]
        )
    rows = [
        {
            "orientation_deg": 30.0,
            "dx_m": case.dx_m,
            "dy_m": case.dy_m,
            "dt_s": case.dt_s,
            "contrast_fraction": channel.REFERENCE_CONTRAST,
            "noise_model": "clean",
            "snr_db": "",
            "attribute": "thdr",
            "sample_space_ap": paired["sample_space"],
            "sampling_aware_ap": paired["sampling_aware"],
            "paired_ap_difference": (
                paired["sampling_aware"] - paired["sample_space"]
            ),
        }
    ]
    write_csv(OUTPUT / "channel_clean_control.csv", rows)
    return rows


def summarize_pairs(
    fault_rows: list[dict[str, object]],
    channel_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    fault_keys = sorted(
        {
            (float(row["throw_m"]), str(row["noise_model"]), str(row["attribute"]))
            for row in fault_rows
        }
    )
    for throw_m, noise_model, attribute in fault_keys:
        values = np.array(
            [
                float(row["paired_ap_difference"])
                for row in fault_rows
                if float(row["throw_m"]) == throw_m
                and row["noise_model"] == noise_model
                and row["attribute"] == attribute
            ]
        )
        low, high = paired_bootstrap_interval(
            values,
            rng=np.random.default_rng(
                20261111 + int(throw_m) + sum(map(ord, attribute + noise_model))
            ),
        )
        summaries.append(
            {
                "experiment": "fault",
                "throw_m": throw_m,
                "noise_model": noise_model,
                "attribute": attribute,
                "realizations": values.size,
                "mean_paired_ap_difference": float(values.mean()),
                "bootstrap_95pct_low": low,
                "bootstrap_95pct_high": high,
            }
        )

    for noise_model in sorted({str(row["noise_model"]) for row in channel_rows}):
        values = np.array(
            [
                float(row["paired_ap_difference"])
                for row in channel_rows
                if row["noise_model"] == noise_model
            ]
        )
        low, high = paired_bootstrap_interval(
            values,
            rng=np.random.default_rng(20261112 + sum(map(ord, noise_model))),
        )
        summaries.append(
            {
                "experiment": "channel",
                "throw_m": "",
                "noise_model": noise_model,
                "attribute": "thdr",
                "realizations": values.size,
                "mean_paired_ap_difference": float(values.mean()),
                "bootstrap_95pct_low": low,
                "bootstrap_95pct_high": high,
            }
        )
    write_csv(OUTPUT / "paired_bootstrap_summary.csv", summaries)
    return summaries


def plot_shape_rmse(rows: list[dict[str, object]]) -> None:
    rows = [
        row
        for row in rows
        if row["reference_definition"] == "implementation_matched_native"
    ]
    cases = [case.name for case in fault.SAMPLING_CASES[1:]]
    labels = ["25 x 12.5", "12.5 x 25", "25 x 25"]
    x = np.arange(len(cases))
    width = 0.10
    colors = ("#4169a1", "#d0743c", "#4d8b6f", "#815a9e")
    figure, axis = plt.subplots(figsize=(9.0, 4.7), constrained_layout=True)
    for attribute_index, (attribute, color) in enumerate(zip(ATTRIBUTES, colors)):
        for mode_index, mode in enumerate(("sample_space", "sampling_aware")):
            values = [
                float(
                    next(
                        row["shape_rmse"]
                        for row in rows
                        if row["sampling_case"] == case
                        and row["implementation"] == mode
                        and row["attribute"] == attribute
                    )
                )
                for case in cases
            ]
            offset = (attribute_index * 2 + mode_index - 3.5) * width
            axis.bar(
                x + offset,
                values,
                width,
                color=color,
                alpha=0.45 if mode == "sample_space" else 1.0,
                hatch="//" if mode == "sample_space" else None,
            )
    axis.set_xticks(x, labels)
    axis.set_xlabel("Target horizontal sampling (m); dt = 4 ms")
    axis.set_ylabel("Shape RMSE")
    axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Patch(facecolor=color, label=attribute.upper())
        for attribute, color in zip(ATTRIBUTES, colors)
    ]
    handles.extend(
        (
            Patch(
                facecolor="0.55",
                alpha=0.45,
                hatch="//",
                label="sample space",
            ),
            Patch(facecolor="0.55", label="sampling aware"),
        )
    )
    axis.legend(handles=handles, frameon=False, ncol=3, fontsize=8)
    figure.savefig(OUTPUT / "figure_3_shape_rmse.png", dpi=600)
    plt.close(figure)


def plot_paired_changes(rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["experiment"] == "fault"]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True, constrained_layout=True)
    colors = dict(zip(ATTRIBUTES, ("#4169a1", "#d0743c", "#4d8b6f", "#815a9e")))
    for axis, noise_model in zip(
        axes, ("gaussian", "colored_impulsive_footprint")
    ):
        for attribute in ATTRIBUTES:
            data = [
                row
                for row in selected
                if row["noise_model"] == noise_model
                and row["attribute"] == attribute
            ]
            data.sort(key=lambda row: float(row["throw_m"]))
            x = np.array([float(row["throw_m"]) for row in data])
            y = np.array([float(row["mean_paired_ap_difference"]) for row in data])
            low = np.array([float(row["bootstrap_95pct_low"]) for row in data])
            high = np.array([float(row["bootstrap_95pct_high"]) for row in data])
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                marker="o",
                capsize=3,
                color=colors[attribute],
                label=attribute.upper(),
            )
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_title(noise_model.replace("_", " "))
        axis.set_xlabel("Fault throw (m)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Paired AP difference (sampling aware - sample space)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(OUTPUT / "figure_4_paired_ap.png", dpi=600)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Run deterministic tables only; skip the 30-realization tests.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Running cross-grid shape-RMSE experiment...", flush=True)
    shape_rows = cross_grid_shape_experiment()
    print("Running velocity-sensitivity experiment...", flush=True)
    velocity_sensitivity_experiment()
    print("Running operator-sensitivity experiment...", flush=True)
    operator_sensitivity_experiment()
    print("Running deterministic clean-channel control...", flush=True)
    clean_channel_control()
    plot_shape_rmse(shape_rows)

    if not args.skip_noise:
        print("Running 30-realization paired fault experiment...", flush=True)
        fault_rows = _paired_fault_rows()
        print("Running 30-realization paired channel experiment...", flush=True)
        channel_rows = _paired_channel_rows()
        summaries = summarize_pairs(fault_rows, channel_rows)
        plot_paired_changes(summaries)
    print(f"Revision results written to {OUTPUT}")


if __name__ == "__main__":
    main()
