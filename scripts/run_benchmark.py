"""Run the independent-grid benchmark proposed for the Short Note."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/seisgrad-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seisgrad.attributes import gradient_attributes
from seisgrad.baselines import unsteered_semblance_discontinuity
from seisgrad.forward import (
    acoustic_impedance,
    depth_mask_to_regular_time,
    depth_property_to_regular_time,
    poststack_synthetic,
)
from seisgrad.metrics import (
    add_gaussian_noise,
    average_precision,
    best_f1,
    correlation,
    normalized_rmse,
)
from seisgrad.model import Grid3D, build_layered_fault_model


@dataclass(frozen=True)
class SamplingCase:
    name: str
    dx_m: float
    dy_m: float
    dt_s: float


EXTENT_X_M = 800.0
EXTENT_Y_M = 800.0
EXTENT_Z_M = 800.0
DZ_M = 5.0
FREQUENCY_HZ = 30.0

SAMPLING_CASES = (
    SamplingCase("reference", 12.5, 12.5, 0.002),
    SamplingCase("coarse_inline", 25.0, 12.5, 0.004),
    SamplingCase("coarse_crossline", 12.5, 25.0, 0.004),
    SamplingCase("coarse_both", 25.0, 25.0, 0.004),
)


def grid_for(case: SamplingCase) -> Grid3D:
    return Grid3D.from_extent(
        extent_x_m=EXTENT_X_M,
        extent_y_m=EXTENT_Y_M,
        extent_z_m=EXTENT_Z_M,
        dx=case.dx_m,
        dy=case.dy_m,
        dz=DZ_M,
    )


def generate_case(
    case: SamplingCase,
    *,
    throw_m: float,
    strike_deg: float = 35.0,
    dip_deg: float = 60.0,
):
    grid = grid_for(case)
    model = build_layered_fault_model(
        grid,
        throw_m=throw_m,
        strike_deg=strike_deg,
        dip_deg=dip_deg,
    )
    seismic, time_s = poststack_synthetic(
        model.vp,
        model.density,
        dz_m=grid.dz,
        dt_s=case.dt_s,
        frequency_hz=FREQUENCY_HZ,
        normalize=False,
    )
    velocity_time = depth_property_to_regular_time(
        model.vp,
        model.vp,
        dz_m=grid.dz,
        time_s=time_s,
    )
    fault_time = depth_mask_to_regular_time(
        model.fault_mask,
        model.vp,
        dz_m=grid.dz,
        time_s=time_s,
        half_width_samples=1,
    )
    return model, seismic, time_s, velocity_time, fault_time


def common_interior(a: np.ndarray, b: np.ndarray, margin: tuple[int, int, int] = (2, 2, 8)):
    shape = tuple(min(left, right) for left, right in zip(a.shape, b.shape))
    slices = tuple(slice(width, size - width) for size, width in zip(shape, margin))
    return a[slices], b[slices]


def score_from_attribute(name: str, values: np.ndarray) -> np.ndarray:
    if name in {"tdr", "tahg"}:
        return np.abs(values)
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    """Read stored benchmark rows for figure-only regeneration."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows: list[dict[str, object]] = list(csv.DictReader(stream))
    for row in rows:
        if "throw_m" in row:
            row["throw_m"] = float(row["throw_m"])
        if "snr_db" in row and row["snr_db"] != "clean":
            row["snr_db"] = float(row["snr_db"])
    return rows


def invariance_experiment(output: Path) -> list[dict[str, object]]:
    reference_case = SAMPLING_CASES[0]
    _, reference_seismic, _, reference_velocity, _ = generate_case(reference_case, throw_m=50.0)
    reference_attributes = {
        mode: gradient_attributes(
            reference_seismic,
            spacing=(reference_case.dx_m, reference_case.dy_m, reference_case.dt_s),
            reference_velocity_m_s=reference_velocity,
            sampling_aware=aware,
        )
        for mode, aware in (("sample_space", False), ("sampling_aware", True))
    }

    rows: list[dict[str, object]] = []
    for case in SAMPLING_CASES[1:]:
        _, seismic, _, velocity, _ = generate_case(case, throw_m=50.0)
        factors = (
            int(round(case.dx_m / reference_case.dx_m)),
            int(round(case.dy_m / reference_case.dy_m)),
            int(round(case.dt_s / reference_case.dt_s)),
        )
        for mode, aware in (("sample_space", False), ("sampling_aware", True)):
            attributes = gradient_attributes(
                seismic,
                spacing=(case.dx_m, case.dy_m, case.dt_s),
                reference_velocity_m_s=velocity,
                sampling_aware=aware,
            )
            for name, values in attributes.items():
                reference = reference_attributes[mode][name][
                    :: factors[0], :: factors[1], :: factors[2]
                ]
                reference, estimate = common_interior(reference, values)
                rows.append(
                    {
                        "sampling_case": case.name,
                        "implementation": mode,
                        "attribute": name,
                        "normalized_rmse": normalized_rmse(reference, estimate),
                        "correlation": correlation(reference, estimate),
                    }
                )
    write_csv(output / "independent_grid_invariance.csv", rows)
    return rows


def detection_experiment(output: Path) -> list[dict[str, object]]:
    case = SAMPLING_CASES[0]
    rows: list[dict[str, object]] = []
    throws_m = (25.0, 50.0, 100.0)
    snr_values: tuple[float | None, ...] = (None, 20.0, 10.0, 5.0)
    noisy_realizations = 5

    for throw_m in throws_m:
        _, clean, _, velocity, fault = generate_case(case, throw_m=throw_m)
        evaluation_slices = (slice(2, -2), slice(2, -2), slice(8, -8))
        labels = fault[evaluation_slices]
        for snr_db in snr_values:
            realization_count = 1 if snr_db is None else noisy_realizations
            for realization in range(realization_count):
                seed = 20260807 + int(throw_m) * 1000 + int(snr_db or 0) * 10 + realization
                noisy = add_gaussian_noise(
                    clean,
                    snr_db,
                    rng=np.random.default_rng(seed),
                )
                for mode, aware in (("sample_space", False), ("sampling_aware", True)):
                    attributes = gradient_attributes(
                        noisy,
                        spacing=(case.dx_m, case.dy_m, case.dt_s),
                        reference_velocity_m_s=velocity,
                        sampling_aware=aware,
                    )
                    for name, values in attributes.items():
                        scores = score_from_attribute(name, values[evaluation_slices])
                        f1, threshold = best_f1(labels, scores)
                        rows.append(
                            {
                                "throw_m": throw_m,
                                "throw_over_wavelength": throw_m / (3000.0 / FREQUENCY_HZ),
                                "snr_db": "clean" if snr_db is None else snr_db,
                                "realization": realization,
                                "implementation": mode,
                                "attribute": name,
                                "average_precision": average_precision(labels, scores),
                                "best_f1": f1,
                                "best_threshold": threshold,
                            }
                        )

                semblance = unsteered_semblance_discontinuity(noisy)
                scores = semblance[evaluation_slices]
                f1, threshold = best_f1(labels, scores)
                rows.append(
                    {
                        "throw_m": throw_m,
                        "throw_over_wavelength": throw_m / (3000.0 / FREQUENCY_HZ),
                        "snr_db": "clean" if snr_db is None else snr_db,
                        "realization": realization,
                        "implementation": "baseline",
                        "attribute": "semblance_discontinuity",
                        "average_precision": average_precision(labels, scores),
                        "best_f1": f1,
                        "best_threshold": threshold,
                    }
                )

    write_csv(output / "fault_detection_benchmark.csv", rows)
    summary_rows: list[dict[str, object]] = []
    keys = sorted(
        {
            (row["throw_m"], str(row["snr_db"]), row["implementation"], row["attribute"])
            for row in rows
        }
    )
    for throw_m, snr_key, implementation, attribute in keys:
        selected = [
            row
            for row in rows
            if row["throw_m"] == throw_m
            and str(row["snr_db"]) == snr_key
            and row["implementation"] == implementation
            and row["attribute"] == attribute
        ]
        summary_rows.append(
            {
                "throw_m": throw_m,
                "throw_over_wavelength": selected[0]["throw_over_wavelength"],
                "snr_db": selected[0]["snr_db"],
                "implementation": implementation,
                "attribute": attribute,
                "realizations": len(selected),
                "average_precision_mean": float(
                    np.mean([float(row["average_precision"]) for row in selected])
                ),
                "average_precision_sd": float(
                    np.std([float(row["average_precision"]) for row in selected], ddof=0)
                ),
                "best_f1_mean": float(np.mean([float(row["best_f1"]) for row in selected])),
                "best_f1_sd": float(np.std([float(row["best_f1"]) for row in selected], ddof=0)),
            }
        )
    write_csv(output / "fault_detection_summary.csv", summary_rows)
    return rows


def control_experiment(output: Path) -> list[dict[str, object]]:
    case = SAMPLING_CASES[0]
    _, seismic, _, velocity, _ = generate_case(case, throw_m=0.0)
    rows: list[dict[str, object]] = []
    for mode, aware in (("sample_space", False), ("sampling_aware", True)):
        attributes = gradient_attributes(
            seismic,
            spacing=(case.dx_m, case.dy_m, case.dt_s),
            reference_velocity_m_s=velocity,
            sampling_aware=aware,
        )
        for name, values in attributes.items():
            scores = score_from_attribute(name, values[2:-2, 2:-2, 8:-8])
            rows.append(
                {
                    "implementation": mode,
                    "attribute": name,
                    "median": float(np.median(scores)),
                    "percentile_95": float(np.percentile(scores, 95)),
                    "percentile_99": float(np.percentile(scores, 99)),
                }
            )
    write_csv(output / "no_fault_control.csv", rows)
    return rows


def velocity_sensitivity_experiment(output: Path) -> list[dict[str, object]]:
    """Test robustness to replacing the known velocity field by scalar values."""

    reference_case = SAMPLING_CASES[0]
    target_case = SAMPLING_CASES[1]
    _, reference_seismic, _, reference_velocity, _ = generate_case(reference_case, throw_m=50.0)
    _, target_seismic, _, target_velocity, _ = generate_case(target_case, throw_m=50.0)
    factors = (
        int(round(target_case.dx_m / reference_case.dx_m)),
        int(round(target_case.dy_m / reference_case.dy_m)),
        int(round(target_case.dt_s / reference_case.dt_s)),
    )
    velocity_cases: tuple[tuple[str, float | None], ...] = (
        ("known_field", None),
        ("scalar_2400", 2400.0),
        ("scalar_3000", 3000.0),
        ("scalar_3600", 3600.0),
    )
    rows: list[dict[str, object]] = []
    for velocity_name, scalar_velocity in velocity_cases:
        reference_v = reference_velocity if scalar_velocity is None else scalar_velocity
        target_v = target_velocity if scalar_velocity is None else scalar_velocity
        reference_attributes = gradient_attributes(
            reference_seismic,
            spacing=(reference_case.dx_m, reference_case.dy_m, reference_case.dt_s),
            reference_velocity_m_s=reference_v,
            sampling_aware=True,
        )
        target_attributes = gradient_attributes(
            target_seismic,
            spacing=(target_case.dx_m, target_case.dy_m, target_case.dt_s),
            reference_velocity_m_s=target_v,
            sampling_aware=True,
        )
        for name, values in target_attributes.items():
            reference = reference_attributes[name][:: factors[0], :: factors[1], :: factors[2]]
            reference, estimate = common_interior(reference, values)
            rows.append(
                {
                    "velocity_case": velocity_name,
                    "attribute": name,
                    "normalized_rmse": normalized_rmse(reference, estimate),
                    "correlation": correlation(reference, estimate),
                }
            )
    write_csv(output / "velocity_sensitivity.csv", rows)
    return rows


def orientation_invariance_experiment(output: Path) -> list[dict[str, object]]:
    """Verify that the main grid result is not unique to one fault orientation."""

    reference_case = SAMPLING_CASES[0]
    target_case = SAMPLING_CASES[1]
    geometries = (
        (0.0, 60.0),
        (30.0, 60.0),
        (60.0, 60.0),
        (90.0, 60.0),
        (35.0, 45.0),
        (35.0, 60.0),
        (35.0, 75.0),
    )
    factors = (
        int(round(target_case.dx_m / reference_case.dx_m)),
        int(round(target_case.dy_m / reference_case.dy_m)),
        int(round(target_case.dt_s / reference_case.dt_s)),
    )
    rows: list[dict[str, object]] = []
    for strike_deg, dip_deg in geometries:
        _, reference_seismic, _, reference_velocity, _ = generate_case(
            reference_case,
            throw_m=50.0,
            strike_deg=strike_deg,
            dip_deg=dip_deg,
        )
        _, target_seismic, _, target_velocity, _ = generate_case(
            target_case,
            throw_m=50.0,
            strike_deg=strike_deg,
            dip_deg=dip_deg,
        )
        for mode, aware in (("sample_space", False), ("sampling_aware", True)):
            reference_attributes = gradient_attributes(
                reference_seismic,
                spacing=(reference_case.dx_m, reference_case.dy_m, reference_case.dt_s),
                reference_velocity_m_s=reference_velocity,
                sampling_aware=aware,
            )
            target_attributes = gradient_attributes(
                target_seismic,
                spacing=(target_case.dx_m, target_case.dy_m, target_case.dt_s),
                reference_velocity_m_s=target_velocity,
                sampling_aware=aware,
            )
            for name, values in target_attributes.items():
                reference = reference_attributes[name][:: factors[0], :: factors[1], :: factors[2]]
                reference, estimate = common_interior(reference, values)
                rows.append(
                    {
                        "strike_deg": strike_deg,
                        "dip_deg": dip_deg,
                        "implementation": mode,
                        "attribute": name,
                        "normalized_rmse": normalized_rmse(reference, estimate),
                        "correlation": correlation(reference, estimate),
                    }
                )
    write_csv(output / "orientation_invariance.csv", rows)
    return rows


def label_panel(axis: plt.Axes, letter: str) -> None:
    """Add a publication-scale subfigure label inside an axis."""

    axis.text(
        0.025,
        0.975,
        f"({letter})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.7},
        zorder=10,
    )


def plot_model(output: Path) -> None:
    case = SAMPLING_CASES[0]
    model, seismic, time_s, _, fault_time = generate_case(case, throw_m=50.0)
    impedance = acoustic_impedance(model.vp, model.density)
    y_index = model.grid.ny // 2
    seismic_limit = float(np.percentile(np.abs(seismic[:, y_index, :]), 99.0))

    figure, axes = plt.subplots(1, 4, figsize=(14.0, 3.8), constrained_layout=True)
    axes[0].imshow(
        model.layer_index[:, y_index, :].T,
        cmap="viridis",
        origin="upper",
        aspect="auto",
        extent=(0, EXTENT_X_M, EXTENT_Z_M, 0),
    )
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("Depth (m)")
    axes[1].imshow(
        impedance[:, y_index, :].T,
        cmap="magma",
        origin="upper",
        aspect="auto",
        extent=(0, EXTENT_X_M, EXTENT_Z_M, 0),
    )
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("Depth (m)")
    axes[2].imshow(
        seismic[:, y_index, :].T,
        cmap="gray",
        origin="upper",
        aspect="auto",
        vmin=-seismic_limit,
        vmax=seismic_limit,
        extent=(0, EXTENT_X_M, time_s[-1], 0),
    )
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("Two-way time (s)")
    axes[3].imshow(
        fault_time[:, y_index, :].T,
        cmap="Reds",
        origin="upper",
        aspect="auto",
        vmin=0,
        vmax=1,
        extent=(0, EXTENT_X_M, time_s[-1], 0),
    )
    axes[3].set_xlabel("x (m)")
    axes[3].set_ylabel("Two-way time (s)")
    for axis, letter in zip(axes, "abcd"):
        label_panel(axis, letter)
    figure.savefig(output / "benchmark_figure_1_model.png", dpi=240)
    plt.close(figure)


def plot_submission_figure_1(output: Path) -> None:
    """Complete the fault benchmark from physical model to noisy attributes."""

    case = SAMPLING_CASES[0]
    model, clean, time_s, velocity, fault_time = generate_case(case, throw_m=50.0)
    impedance = acoustic_impedance(model.vp, model.density)
    seed = 20260807 + 50 * 1000 + 10 * 10
    noisy = add_gaussian_noise(clean, 10.0, rng=np.random.default_rng(seed))
    attributes = gradient_attributes(
        noisy,
        spacing=(case.dx_m, case.dy_m, case.dt_s),
        reference_velocity_m_s=velocity,
        sampling_aware=True,
    )
    semblance = unsteered_semblance_discontinuity(noisy)
    y_index = model.grid.ny // 2
    fault_section = fault_time[:, y_index, :]
    trace_x: list[float] = []
    trace_t: list[float] = []
    for ix in range(fault_section.shape[0]):
        time_indices = np.flatnonzero(fault_section[ix])
        if time_indices.size:
            trace_x.append(float(model.grid.x[ix]))
            trace_t.append(float(np.median(time_s[time_indices])))

    clean_section = clean[:, y_index, :].T
    noisy_section = noisy[:, y_index, :].T
    seismic_limit = float(
        max(
            np.percentile(np.abs(clean_section[8:-8, 2:-2]), 99.0),
            np.percentile(np.abs(noisy_section[8:-8, 2:-2]), 99.0),
        )
    )
    panels = (
        (
            "impedance",
            impedance[:, y_index, :].T,
            "viridis",
            "Impedance (kg m$^{-2}$ s$^{-1}$)",
        ),
        ("clean", clean_section, "gray", "Amplitude"),
        ("noisy", noisy_section, "gray", "Amplitude"),
        (
            "thdr",
            attributes["thdr"][:, y_index, :].T,
            "magma",
            "Amplitude m$^{-1}$",
        ),
        (
            "asag",
            attributes["asag"][:, y_index, :].T,
            "magma",
            "Amplitude m$^{-1}$",
        ),
        (
            "semblance",
            semblance[:, y_index, :].T,
            "magma",
            "Discontinuity",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.8, 7.1), constrained_layout=True)
    for axis, (name, values, cmap, colorbar_label), letter in zip(
        axes.ravel(), panels, "abcdef"
    ):
        if name == "impedance":
            image = axis.imshow(
                values,
                cmap=cmap,
                origin="upper",
                aspect="auto",
                extent=(0, EXTENT_X_M, EXTENT_Z_M, 0),
            )
            axis.set_ylabel("Depth (m)")
        else:
            if name in {"clean", "noisy"}:
                vmin, vmax = -seismic_limit, seismic_limit
            elif name in {"thdr", "asag", "semblance"}:
                vmin = 0.0
                vmax = float(np.percentile(values[8:-8, 2:-2], 99.0))
            else:
                vmin, vmax = -0.5 * np.pi, 0.5 * np.pi
            image = axis.imshow(
                values,
                cmap=cmap,
                origin="upper",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=(0, EXTENT_X_M, time_s[-1], 0),
                interpolation="nearest",
            )
            axis.plot(trace_x, trace_t, color="black", linewidth=1.8, zorder=7)
            axis.plot(
                trace_x,
                trace_t,
                color="white",
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=8,
            )
            axis.set_ylabel("Two-way time (s)")
        axis.set_xlabel("x (m)")
        axis.set_xticks((0, 200, 400, 600, 800))
        colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.022)
        colorbar.set_label(colorbar_label)
        label_panel(axis, letter)
    figure.savefig(output / "submission_figure_1_fault_complete.png", dpi=600)
    plt.close(figure)


def plot_invariance(output: Path, rows: list[dict[str, object]]) -> None:
    attributes = ("thdr", "asag", "tdr", "tahg")
    cases = tuple(case.name for case in SAMPLING_CASES[1:])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    colors = {"sample_space": "#c65d3a", "sampling_aware": "#2f7d8c"}
    for axis, metric, ylabel in (
        (axes[0], "normalized_rmse", "Normalized RMSE"),
        (axes[1], "correlation", "Correlation"),
    ):
        x = np.arange(len(attributes))
        width = 0.36
        for offset, mode in ((-width / 2, "sample_space"), (width / 2, "sampling_aware")):
            values = []
            for attribute in attributes:
                selected = [
                    float(row[metric])
                    for row in rows
                    if row["implementation"] == mode
                    and row["attribute"] == attribute
                    and row["sampling_case"] in cases
                ]
                values.append(float(np.mean(selected)))
            axis.bar(x + offset, values, width, label=mode.replace("_", " "), color=colors[mode])
        axis.set_xticks(x, [name.upper() for name in attributes])
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    for axis, letter in zip(axes, "ab"):
        label_panel(axis, letter)
    figure.savefig(output / "benchmark_figure_2_invariance.png", dpi=600)
    plt.close(figure)


def plot_attribute_horizontal_slice(output: Path) -> None:
    """Plot a supplementary horizontal section of all four attributes."""

    case = SAMPLING_CASES[0]
    model, seismic, time_s, velocity, fault_time = generate_case(case, throw_m=50.0)
    attributes = gradient_attributes(
        seismic,
        spacing=(case.dx_m, case.dy_m, case.dt_s),
        reference_velocity_m_s=velocity,
        sampling_aware=True,
    )
    time_index = int(np.argmin(np.abs(time_s - 0.320)))
    fault_slice = fault_time[:, :, time_index]
    trace_x: list[float] = []
    trace_y: list[float] = []
    for ix in range(fault_slice.shape[0]):
        y_indices = np.flatnonzero(fault_slice[ix])
        if y_indices.size:
            trace_x.append(float(model.grid.x[ix]))
            trace_y.append(float(np.median(model.grid.y[y_indices])))

    panels = (
        ("thdr", "magma", "Amplitude m$^{-1}$"),
        ("asag", "magma", "Amplitude m$^{-1}$"),
        ("tdr", "RdBu_r", "Radians"),
        ("tahg", "RdBu_r", "Radians"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 7.5), constrained_layout=True)
    for axis, (name, cmap, colorbar_label), letter in zip(
        axes.ravel(), panels, "abcd"
    ):
        values = attributes[name][:, :, time_index].T
        if name in {"thdr", "asag"}:
            display_limit = float(np.percentile(values[2:-2, 2:-2], 99.0))
            image = axis.imshow(
                values,
                origin="lower",
                extent=(0.0, EXTENT_X_M, 0.0, EXTENT_Y_M),
                cmap=cmap,
                vmin=0.0,
                vmax=display_limit,
                interpolation="nearest",
            )
        else:
            image = axis.imshow(
                values,
                origin="lower",
                extent=(0.0, EXTENT_X_M, 0.0, EXTENT_Y_M),
                cmap=cmap,
                vmin=-0.5 * np.pi,
                vmax=0.5 * np.pi,
                interpolation="nearest",
            )
        axis.plot(trace_x, trace_y, color="black", linewidth=1.8, zorder=7)
        axis.plot(
            trace_x,
            trace_y,
            color="white",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=8,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        axis.set_xticks((0, 200, 400, 600, 800))
        axis.set_yticks((0, 200, 400, 600, 800))
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86, pad=0.025)
        colorbar.set_label(colorbar_label)
        label_panel(axis, letter)
    figure.savefig(output / "supplement_attribute_slice_horizontal.png", dpi=240)
    plt.close(figure)


def plot_attribute_vertical_section(output: Path) -> None:
    """Plot all four attributes on the same vertical section as Figure 1."""

    case = SAMPLING_CASES[0]
    model, seismic, time_s, velocity, fault_time = generate_case(case, throw_m=50.0)
    attributes = gradient_attributes(
        seismic,
        spacing=(case.dx_m, case.dy_m, case.dt_s),
        reference_velocity_m_s=velocity,
        sampling_aware=True,
    )
    y_index = model.grid.ny // 2

    fault_section = fault_time[:, y_index, :]
    trace_x: list[float] = []
    trace_t: list[float] = []
    for ix in range(fault_section.shape[0]):
        time_indices = np.flatnonzero(fault_section[ix])
        if time_indices.size:
            trace_x.append(float(model.grid.x[ix]))
            trace_t.append(float(np.median(time_s[time_indices])))

    panels = (
        ("thdr", "magma", "Amplitude m$^{-1}$"),
        ("asag", "magma", "Amplitude m$^{-1}$"),
        ("tdr", "RdBu_r", "Radians"),
        ("tahg", "RdBu_r", "Radians"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 7.2), constrained_layout=True)
    for axis, (name, cmap, colorbar_label), letter in zip(
        axes.ravel(), panels, "abcd"
    ):
        values = attributes[name][:, y_index, :].T
        if name in {"thdr", "asag"}:
            display_limit = float(np.percentile(values[8:-8, 2:-2], 99.0))
            image = axis.imshow(
                values,
                origin="upper",
                extent=(0.0, EXTENT_X_M, time_s[-1], 0.0),
                aspect="auto",
                cmap=cmap,
                vmin=0.0,
                vmax=display_limit,
                interpolation="nearest",
            )
        else:
            image = axis.imshow(
                values,
                origin="upper",
                extent=(0.0, EXTENT_X_M, time_s[-1], 0.0),
                aspect="auto",
                cmap=cmap,
                vmin=-0.5 * np.pi,
                vmax=0.5 * np.pi,
                interpolation="nearest",
            )
        axis.plot(trace_x, trace_t, color="black", linewidth=1.8, zorder=7)
        axis.plot(
            trace_x,
            trace_t,
            color="white",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=8,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("Two-way time (s)")
        axis.set_xticks((0, 200, 400, 600, 800))
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86, pad=0.025)
        colorbar.set_label(colorbar_label)
        label_panel(axis, letter)
    figure.savefig(output / "benchmark_figure_3_attribute_section.png", dpi=240)
    plt.close(figure)


def plot_detection(output: Path, rows: list[dict[str, object]]) -> None:
    selected_series = (
        ("sample_space", "thdr", "THDR (both formulations)", "#c65d3a", "-"),
        ("sample_space", "asag", "ASAG sample-space", "#2f7d8c", "--"),
        ("sampling_aware", "asag", "ASAG sampling-aware", "#2f7d8c", "-"),
        ("baseline", "semblance_discontinuity", "Semblance baseline", "#595959", ":"),
    )
    snr_order = ("clean", 20.0, 10.0, 5.0)
    x = np.arange(len(snr_order))
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True, constrained_layout=True)
    for axis, throw_m, letter in zip(axes, (25.0, 50.0, 100.0), "abc"):
        for mode, attribute, label, color, linestyle in selected_series:
            values = []
            uncertainties = []
            for snr in snr_order:
                selected = [
                    float(item["average_precision"])
                    for item in rows
                    if item["throw_m"] == throw_m
                    and item["snr_db"] == snr
                    and item["implementation"] == mode
                    and item["attribute"] == attribute
                ]
                values.append(float(np.mean(selected)))
                uncertainties.append(float(np.std(selected, ddof=0)))
            axis.errorbar(
                x,
                values,
                yerr=uncertainties,
                marker="o",
                capsize=2.5,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        axis.set_xticks(x, ["clean", "20", "10", "5"])
        axis.set_xlabel("SNR (dB)")
        axis.spines[["top", "right"]].set_visible(False)
        label_panel(axis, letter)
    axes[0].set_ylabel("Average precision")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    figure.savefig(output / "benchmark_figure_3_detection.png", dpi=240)
    plt.close(figure)


def plot_submission_figure_4(
    output: Path,
    fault_rows: list[dict[str, object]],
    channel_rows: list[dict[str, object]],
) -> None:
    """Join fault-noise robustness and channel-orientation sensitivity."""

    selected_series = (
        ("sample_space", "thdr", "THDR", "#c65d3a", "-"),
        ("sample_space", "asag", "ASAG sample-space", "#2f7d8c", "--"),
        ("sampling_aware", "asag", "ASAG sampling-aware", "#2f7d8c", "-"),
        ("baseline", "semblance_discontinuity", "Semblance", "#595959", ":"),
    )
    snr_order = ("clean", 20.0, 10.0, 5.0)
    x_snr = np.arange(len(snr_order))
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 8.0), constrained_layout=True)
    for axis, throw_m, letter in zip(axes[0], (25.0, 100.0), "ab"):
        for mode, attribute, label, color, linestyle in selected_series:
            means = []
            uncertainties = []
            for snr in snr_order:
                selected = [
                    float(item["average_precision"])
                    for item in fault_rows
                    if item["throw_m"] == throw_m
                    and item["snr_db"] == snr
                    and item["implementation"] == mode
                    and item["attribute"] == attribute
                ]
                means.append(float(np.mean(selected)))
                uncertainties.append(float(np.std(selected, ddof=0)))
            axis.errorbar(
                x_snr,
                means,
                yerr=uncertainties,
                marker="o",
                capsize=2.5,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        axis.set_xticks(x_snr, ["clean", "20", "10", "5"])
        axis.set_xlabel("SNR (dB)")
        axis.set_ylim(0.0, 1.0)
        axis.spines[["top", "right"]].set_visible(False)
        label_panel(axis, letter)
    axes[0, 0].set_ylabel("Fault average precision")
    axes[0, 1].legend(frameon=False, fontsize=8, loc="best")

    colors = {"sample_space": "#c65d3a", "sampling_aware": "#2f7d8c"}
    for axis, case, letter in zip(
        axes[1], ("coarse_inline", "coarse_crossline"), "cd"
    ):
        for mode in ("sample_space", "sampling_aware"):
            selected = sorted(
                (
                    (float(row["orientation_deg"]), float(row["average_precision"]))
                    for row in channel_rows
                    if row["sampling_case"] == case
                    and row["implementation"] == mode
                    and row["attribute"] == "thdr"
                ),
                key=lambda pair: pair[0],
            )
            axis.plot(
                [pair[0] for pair in selected],
                [pair[1] for pair in selected],
                marker="o",
                linewidth=1.7,
                markersize=4.5,
                color=colors[mode],
                label=mode.replace("_", " "),
            )
        axis.set_xlabel("Channel orientation (degrees)")
        axis.set_xticks((0, 30, 60, 90, 120, 150))
        axis.set_ylim(0.60, 1.00)
        axis.spines[["top", "right"]].set_visible(False)
        label_panel(axis, letter)
    axes[1, 0].set_ylabel("THDR channel-margin average precision")
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.savefig(output / "submission_figure_4_joint_metrics.png", dpi=600)
    plt.close(figure)


def plot_velocity_sensitivity(output: Path, rows: list[dict[str, object]]) -> None:
    velocity_cases = ("known_field", "scalar_2400", "scalar_3000", "scalar_3600")
    attributes = ("thdr", "asag", "tdr", "tahg")
    x = np.arange(len(velocity_cases))
    colors = ("#c65d3a", "#2f7d8c", "#5b4b8a", "#6c8e3f")
    figure, axis = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    for attribute, color in zip(attributes, colors):
        values = [
            float(
                next(
                    row["normalized_rmse"]
                    for row in rows
                    if row["velocity_case"] == velocity_case and row["attribute"] == attribute
                )
            )
            for velocity_case in velocity_cases
        ]
        axis.plot(x, values, marker="o", label=attribute.upper(), color=color)
    axis.set_xticks(x, ["known field", "2400", "3000", "3600"])
    axis.set_xlabel("Velocity used for time-to-depth derivative (m/s)")
    axis.set_ylabel("Normalized RMSE")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output / "supplement_velocity_sensitivity.png", dpi=240)
    plt.close(figure)


def plot_orientation_invariance(output: Path, rows: list[dict[str, object]]) -> None:
    attributes = ("thdr", "asag", "tdr", "tahg")
    x = np.arange(len(attributes))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    for offset, mode, color in (
        (-width / 2, "sample_space", "#c65d3a"),
        (width / 2, "sampling_aware", "#2f7d8c"),
    ):
        means = []
        standard_deviations = []
        for attribute in attributes:
            values = np.array(
                [
                    float(row["normalized_rmse"])
                    for row in rows
                    if row["implementation"] == mode and row["attribute"] == attribute
                ]
            )
            means.append(float(values.mean()))
            standard_deviations.append(float(values.std(ddof=1)))
        axis.bar(
            x + offset,
            means,
            width,
            yerr=standard_deviations,
            capsize=3,
            label=mode.replace("_", " "),
            color=color,
        )
    axis.set_xticks(x, [name.upper() for name in attributes])
    axis.set_ylabel("Normalized RMSE")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(output / "supplement_orientation_invariance.png", dpi=240)
    plt.close(figure)


def render_figures(output: Path) -> None:
    """Regenerate figures from the stored benchmark tables."""

    invariance_rows = read_csv(output / "independent_grid_invariance.csv")
    detection_rows = read_csv(output / "fault_detection_benchmark.csv")
    velocity_rows = read_csv(output / "velocity_sensitivity.csv")
    orientation_rows = read_csv(output / "orientation_invariance.csv")
    channel_rows = read_csv(ROOT / "output" / "channel_benchmark_v0.1" / "channel_margin_recovery.csv")
    plot_model(output)
    plot_submission_figure_1(output)
    plot_invariance(output, invariance_rows)
    plot_attribute_vertical_section(output)
    plot_attribute_horizontal_slice(output)
    plot_detection(output, detection_rows)
    plot_submission_figure_4(output, detection_rows, channel_rows)
    plot_velocity_sensitivity(output, velocity_rows)
    plot_orientation_invariance(output, orientation_rows)


def main() -> None:
    output = ROOT / "output" / "benchmark_v0.2"
    output.mkdir(parents=True, exist_ok=True)
    if "--figures-only" in sys.argv[1:]:
        print("Rendering figures from stored benchmark tables...")
        render_figures(output)
        print(f"Figures written to {output}")
        return
    print("Running independent-grid invariance experiment...")
    invariance_rows = invariance_experiment(output)
    print("Running fault-detection experiment...")
    detection_rows = detection_experiment(output)
    print("Running no-fault control...")
    control_experiment(output)
    print("Running velocity-sensitivity experiment...")
    velocity_rows = velocity_sensitivity_experiment(output)
    print("Running fault-orientation experiment...")
    orientation_rows = orientation_invariance_experiment(output)
    print("Rendering figures...")
    plot_model(output)
    plot_submission_figure_1(output)
    plot_invariance(output, invariance_rows)
    plot_attribute_vertical_section(output)
    plot_attribute_horizontal_slice(output)
    plot_detection(output, detection_rows)
    channel_rows = read_csv(ROOT / "output" / "channel_benchmark_v0.1" / "channel_margin_recovery.csv")
    plot_submission_figure_4(output, detection_rows, channel_rows)
    plot_velocity_sensitivity(output, velocity_rows)
    plot_orientation_invariance(output, orientation_rows)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
