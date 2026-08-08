"""Exploratory channel benchmark for the sampling-aware Short Note.

The experiment asks two separate questions:

1. Does physical-coordinate differentiation improve cross-grid invariance for
   a planform geological target whose margins span many azimuths?
2. Do the resulting attributes recover channel margins more usefully than the
   input amplitude or a simple semblance-discontinuity baseline?

The channel experiment complements the three-dimensional fault benchmark with
a planform target whose margins span a broad range of azimuths.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/seisgrad-channel-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seisgrad.attributes import gradient_attributes
from seisgrad.baselines import unsteered_semblance_discontinuity
from seisgrad.forward import acoustic_impedance, depth_property_to_regular_time, poststack_synthetic
from seisgrad.metrics import add_gaussian_noise, average_precision, best_f1, correlation, normalized_rmse
from seisgrad.model import Grid3D, IncisedChannelModel, build_incised_channel_model


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
ORIENTATIONS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
REFERENCE_CONTRAST = 0.08
MARGIN_TOLERANCE_M = 20.0

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


def top_time_indices(model: IncisedChannelModel, time_s: np.ndarray) -> np.ndarray:
    """Return the regular-time sample nearest the channel-top horizon."""

    iz = int(round(model.channel_top_depth_m / model.grid.dz))
    layer_dt = 2.0 * model.grid.dz / np.maximum(model.vp, 1.0)
    twt = np.cumsum(layer_dt, axis=-1)
    twt -= twt[..., :1]
    indices = np.rint(twt[..., iz] / float(np.median(np.diff(time_s)))).astype(int)
    return np.clip(indices, 0, time_s.size - 1)


def generate_case(
    case: SamplingCase,
    *,
    orientation_deg: float,
    contrast_fraction: float,
) -> tuple[IncisedChannelModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = grid_for(case)
    model = build_incised_channel_model(
        grid,
        orientation_deg=orientation_deg,
        impedance_contrast_fraction=contrast_fraction,
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
    return model, seismic, time_s, velocity_time, top_time_indices(model, time_s)


def surface_samples(volume: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(volume, indices[..., None], axis=-1)[..., 0]


def surface_window_absmax(
    volume: np.ndarray,
    indices: np.ndarray,
    *,
    half_width_samples: int,
) -> np.ndarray:
    offsets = np.arange(-half_width_samples, half_width_samples + 1)
    gather = np.clip(indices[..., None] + offsets, 0, volume.shape[-1] - 1)
    values = np.take_along_axis(volume, gather, axis=-1)
    return np.max(np.abs(values), axis=-1)


def evaluation_masks(model: IncisedChannelModel) -> tuple[np.ndarray, np.ndarray]:
    x2, y2 = np.meshgrid(model.grid.x, model.grid.y, indexing="ij")
    edge = (
        (x2 >= 50.0)
        & (x2 <= EXTENT_X_M - 50.0)
        & (y2 >= 50.0)
        & (y2 <= EXTENT_Y_M - 50.0)
    )
    labels = np.abs(model.channel_signed_margin_distance_m) <= MARGIN_TOLERANCE_M
    return labels & edge, edge


def score_maps(
    seismic: np.ndarray,
    attributes: dict[str, np.ndarray],
    semblance: np.ndarray,
    top_indices: np.ndarray,
    dt_s: float,
) -> dict[str, np.ndarray]:
    half_width = max(1, int(round(0.006 / dt_s)))
    seismic_surface = surface_samples(seismic, top_indices)
    amplitude_anomaly = np.abs(seismic_surface - np.median(seismic_surface))
    thdr = surface_window_absmax(attributes["thdr"], top_indices, half_width_samples=half_width)
    asag = surface_window_absmax(attributes["asag"], top_indices, half_width_samples=half_width)
    tdr_magnitude = surface_window_absmax(
        attributes["tdr"], top_indices, half_width_samples=half_width
    )
    tahg_magnitude = surface_window_absmax(
        attributes["tahg"], top_indices, half_width_samples=half_width
    )
    semblance_score = surface_window_absmax(
        semblance, top_indices, half_width_samples=half_width
    )

    # Exploratory, explicitly conditioned interpretation of the TDR zero
    # contour.  It is reported separately from the raw four-attribute test.
    tdr_surface = surface_samples(attributes["tdr"], top_indices)
    thdr_scale = max(float(np.percentile(thdr, 99.0)), 1e-12)
    tdr_zero_conditioned = np.exp(-((tdr_surface / 0.30) ** 2)) * np.clip(
        thdr / thdr_scale, 0.0, 1.0
    )
    return {
        "input_amplitude": amplitude_anomaly,
        "thdr": thdr,
        "asag": asag,
        "tdr_magnitude": tdr_magnitude,
        "tahg_magnitude": tahg_magnitude,
        "tdr_zero_conditioned": tdr_zero_conditioned,
        "semblance_discontinuity": semblance_score,
    }


def score_recovery(
    labels: np.ndarray,
    evaluation: np.ndarray,
    scores: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    output: dict[str, tuple[float, float]] = {}
    for name, values in scores.items():
        f1, _ = best_f1(labels[evaluation], values[evaluation])
        output[name] = (average_precision(labels[evaluation], values[evaluation]), f1)
    return output


def crop_matching(
    reference: np.ndarray,
    estimate: np.ndarray,
    margin: tuple[int, int, int] = (2, 2, 2),
) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(min(left, right) for left, right in zip(reference.shape, estimate.shape))
    slices = tuple(slice(width, size - width) for size, width in zip(shape, margin))
    return reference[slices], estimate[slices]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def invariance_and_orientation_experiment(
    output: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reference_case = SAMPLING_CASES[0]
    invariance_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []

    for orientation in ORIENTATIONS_DEG:
        reference_model, reference_seismic, reference_time, reference_velocity, reference_top = (
            generate_case(
                reference_case,
                orientation_deg=orientation,
                contrast_fraction=REFERENCE_CONTRAST,
            )
        )
        reference_attributes = {
            mode: gradient_attributes(
                reference_seismic,
                spacing=(reference_case.dx_m, reference_case.dy_m, reference_case.dt_s),
                reference_velocity_m_s=reference_velocity,
                sampling_aware=aware,
            )
            for mode, aware in (("sample_space", False), ("sampling_aware", True))
        }

        reference_semblance = unsteered_semblance_discontinuity(reference_seismic)
        reference_labels, reference_eval = evaluation_masks(reference_model)
        for mode, attributes in reference_attributes.items():
            scores = score_maps(
                reference_seismic,
                attributes,
                reference_semblance,
                reference_top,
                reference_case.dt_s,
            )
            recovery = score_recovery(reference_labels, reference_eval, scores)
            for attribute, (ap, f1) in recovery.items():
                recovery_rows.append(
                    {
                        "sampling_case": reference_case.name,
                        "orientation_deg": orientation,
                        "implementation": mode if attribute not in {"input_amplitude", "semblance_discontinuity"} else "baseline",
                        "attribute": attribute,
                        "average_precision": ap,
                        "best_f1": f1,
                    }
                )

        for case in SAMPLING_CASES[1:]:
            model, seismic, time_s, velocity, top = generate_case(
                case,
                orientation_deg=orientation,
                contrast_fraction=REFERENCE_CONTRAST,
            )
            factors = (
                int(round(case.dx_m / reference_case.dx_m)),
                int(round(case.dy_m / reference_case.dy_m)),
                int(round(case.dt_s / reference_case.dt_s)),
            )
            target_attributes = {
                mode: gradient_attributes(
                    seismic,
                    spacing=(case.dx_m, case.dy_m, case.dt_s),
                    reference_velocity_m_s=velocity,
                    sampling_aware=aware,
                )
                for mode, aware in (("sample_space", False), ("sampling_aware", True))
            }

            target_semblance = unsteered_semblance_discontinuity(seismic)
            labels, evaluation = evaluation_masks(model)
            for mode, attributes in target_attributes.items():
                scores = score_maps(seismic, attributes, target_semblance, top, case.dt_s)
                recovery = score_recovery(labels, evaluation, scores)
                for attribute, (ap, f1) in recovery.items():
                    recovery_rows.append(
                        {
                            "sampling_case": case.name,
                            "orientation_deg": orientation,
                            "implementation": mode if attribute not in {"input_amplitude", "semblance_discontinuity"} else "baseline",
                            "attribute": attribute,
                            "average_precision": ap,
                            "best_f1": f1,
                        }
                    )

                for name, values in attributes.items():
                    reference = reference_attributes[mode][name][
                        :: factors[0], :: factors[1], :: factors[2]
                    ]
                    estimate = values
                    # Limit comparison to a channel-centred temporal window.
                    top_t = float(np.median(reference_time[reference_top]))
                    t0 = max(0.0, top_t - 0.040)
                    t1 = min(float(reference_time[-1]), top_t + 0.080)
                    reference_ds_time = reference_time[:: factors[2]]
                    target_time_mask = (time_s >= t0) & (time_s <= t1)
                    reference_time_mask = (reference_ds_time >= t0) & (reference_ds_time <= t1)
                    reference = reference[..., reference_time_mask]
                    estimate = estimate[..., target_time_mask]
                    reference, estimate = crop_matching(reference, estimate)
                    invariance_rows.append(
                        {
                            "sampling_case": case.name,
                            "orientation_deg": orientation,
                            "implementation": mode,
                            "attribute": name,
                            "normalized_rmse": normalized_rmse(reference, estimate),
                            "correlation": correlation(reference, estimate),
                        }
                    )

    write_csv(output / "channel_grid_invariance.csv", invariance_rows)

    # Baseline rows are identical for the two derivative modes; remove exact
    # duplicates introduced by the shared scoring loop.
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in recovery_rows:
        key = (
            row["sampling_case"],
            row["orientation_deg"],
            row["implementation"],
            row["attribute"],
        )
        unique[key] = row
    recovery_rows = list(unique.values())
    write_csv(output / "channel_margin_recovery.csv", recovery_rows)
    return invariance_rows, recovery_rows


def contrast_noise_experiment(output: Path) -> list[dict[str, object]]:
    case = SAMPLING_CASES[0]
    rows: list[dict[str, object]] = []
    for contrast in (0.03, 0.06, 0.12):
        model, clean, _, velocity, top = generate_case(
            case,
            orientation_deg=30.0,
            contrast_fraction=contrast,
        )
        labels, evaluation = evaluation_masks(model)
        for snr_db in (None, 10.0):
            noisy = add_gaussian_noise(
                clean,
                snr_db,
                rng=np.random.default_rng(20260808 + int(contrast * 1000) + int(snr_db or 0)),
            )
            semblance = unsteered_semblance_discontinuity(noisy)
            for mode, aware in (("sample_space", False), ("sampling_aware", True)):
                attributes = gradient_attributes(
                    noisy,
                    spacing=(case.dx_m, case.dy_m, case.dt_s),
                    reference_velocity_m_s=velocity,
                    sampling_aware=aware,
                )
                recovery = score_recovery(
                    labels,
                    evaluation,
                    score_maps(noisy, attributes, semblance, top, case.dt_s),
                )
                for attribute, (ap, f1) in recovery.items():
                    rows.append(
                        {
                            "contrast_fraction": contrast,
                            "snr_db": "clean" if snr_db is None else snr_db,
                            "implementation": mode if attribute not in {"input_amplitude", "semblance_discontinuity"} else "baseline",
                            "attribute": attribute,
                            "average_precision": ap,
                            "best_f1": f1,
                        }
                    )
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["contrast_fraction"],
            row["snr_db"],
            row["implementation"],
            row["attribute"],
        )
        unique[key] = row
    rows = list(unique.values())
    write_csv(output / "channel_contrast_noise.csv", rows)
    return rows


def label_panel(axis: plt.Axes, letter: str) -> None:
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


def margin_contours(model: IncisedChannelModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return model.grid.x, model.grid.y, model.channel_signed_margin_distance_m.T


def plot_channel_model(output: Path) -> None:
    case = SAMPLING_CASES[0]
    model, seismic, time_s, _, top = generate_case(
        case,
        orientation_deg=30.0,
        contrast_fraction=REFERENCE_CONTRAST,
    )
    impedance = acoustic_impedance(model.vp, model.density)
    y_index = model.grid.ny // 2
    seismic_surface = surface_samples(seismic, top)
    seismic_limit = float(np.percentile(np.abs(seismic_surface), 99.0))

    figure, axes = plt.subplots(2, 2, figsize=(8.8, 7.4), constrained_layout=True)
    axes[0, 0].imshow(
        model.channel_footprint.T,
        origin="lower",
        extent=(0, EXTENT_X_M, 0, EXTENT_Y_M),
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].set_aspect("equal")
    axes[0, 1].imshow(
        impedance[:, y_index, :].T,
        origin="upper",
        extent=(0, EXTENT_X_M, EXTENT_Z_M, 0),
        aspect="auto",
        cmap="viridis",
    )
    axes[0, 1].set_xlabel("x (m)")
    axes[0, 1].set_ylabel("Depth (m)")
    image = axes[1, 0].imshow(
        seismic_surface.T,
        origin="lower",
        extent=(0, EXTENT_X_M, 0, EXTENT_Y_M),
        aspect="equal",
        cmap="gray",
        vmin=-seismic_limit,
        vmax=seismic_limit,
    )
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 0].set_ylabel("y (m)")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.82, label="Amplitude")
    axes[1, 1].imshow(
        np.abs(model.channel_signed_margin_distance_m.T) <= MARGIN_TOLERANCE_M,
        origin="lower",
        extent=(0, EXTENT_X_M, 0, EXTENT_Y_M),
        cmap="Reds",
        vmin=0,
        vmax=1,
        aspect="equal",
    )
    axes[1, 1].set_xlabel("x (m)")
    axes[1, 1].set_ylabel("y (m)")
    for axis, letter in zip(axes.ravel(), "abcd"):
        label_panel(axis, letter)
    figure.savefig(output / "channel_model_and_input.png", dpi=240)
    plt.close(figure)


def plot_attribute_maps(output: Path) -> None:
    case = SAMPLING_CASES[0]
    model, seismic, _, velocity, top = generate_case(
        case,
        orientation_deg=30.0,
        contrast_fraction=REFERENCE_CONTRAST,
    )
    attributes = gradient_attributes(
        seismic,
        spacing=(case.dx_m, case.dy_m, case.dt_s),
        reference_velocity_m_s=velocity,
        sampling_aware=True,
    )
    semblance = unsteered_semblance_discontinuity(seismic)
    panels = (
        ("input", surface_samples(seismic, top), "gray", "Amplitude"),
        ("thdr", surface_samples(attributes["thdr"], top), "magma", "Amplitude m$^{-1}$"),
        ("asag", surface_samples(attributes["asag"], top), "magma", "Amplitude m$^{-1}$"),
        ("tdr", surface_samples(attributes["tdr"], top), "RdBu_r", "Radians"),
        ("tahg", surface_samples(attributes["tahg"], top), "RdBu_r", "Radians"),
        ("semblance", surface_samples(semblance, top), "magma", "Discontinuity"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.5), constrained_layout=True)
    x, y, margin = margin_contours(model)
    for axis, (name, values, cmap, colorbar_label), letter in zip(
        axes.ravel(), panels, "abcdef"
    ):
        values = values.T
        if name in {"input", "tdr", "tahg"}:
            limit = (
                0.5 * np.pi
                if name in {"tdr", "tahg"}
                else float(np.percentile(np.abs(values), 99.0))
            )
            image = axis.imshow(
                values,
                origin="lower",
                extent=(0, EXTENT_X_M, 0, EXTENT_Y_M),
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
        else:
            limit = float(np.percentile(values, 99.0))
            image = axis.imshow(
                values,
                origin="lower",
                extent=(0, EXTENT_X_M, 0, EXTENT_Y_M),
                cmap=cmap,
                vmin=0.0,
                vmax=limit,
                interpolation="nearest",
            )
        axis.contour(x, y, margin, levels=[0.0], colors="white", linewidths=1.0, linestyles="--")
        axis.contour(x, y, margin, levels=[0.0], colors="black", linewidths=0.35)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis, shrink=0.77, pad=0.02, label=colorbar_label)
        label_panel(axis, letter)
    figure.savefig(output / "channel_attribute_horizon_slices.png", dpi=240)
    plt.close(figure)


def plot_manuscript_channel_comparison(output: Path) -> None:
    """Render the compact channel figure proposed for the Short Note."""

    case = next(item for item in SAMPLING_CASES if item.name == "coarse_inline")
    model, seismic, _, velocity, top = generate_case(
        case,
        orientation_deg=30.0,
        contrast_fraction=REFERENCE_CONTRAST,
    )
    sample = gradient_attributes(
        seismic,
        spacing=(case.dx_m, case.dy_m, case.dt_s),
        reference_velocity_m_s=velocity,
        sampling_aware=False,
    )
    aware = gradient_attributes(
        seismic,
        spacing=(case.dx_m, case.dy_m, case.dt_s),
        reference_velocity_m_s=velocity,
        sampling_aware=True,
    )
    semblance = unsteered_semblance_discontinuity(seismic)
    half_width = max(1, int(round(0.006 / case.dt_s)))
    panels = (
        (
            "footprint",
            model.channel_footprint.astype(float),
            "Greys",
            None,
        ),
        (
            "input",
            surface_samples(seismic, top),
            "gray",
            "Amplitude",
        ),
        (
            "sample_thdr",
            surface_window_absmax(sample["thdr"], top, half_width_samples=half_width),
            "magma",
            "Amplitude sample$^{-1}$",
        ),
        (
            "aware_thdr",
            surface_window_absmax(aware["thdr"], top, half_width_samples=half_width),
            "magma",
            "Amplitude m$^{-1}$",
        ),
        (
            "aware_asag",
            surface_window_absmax(aware["asag"], top, half_width_samples=half_width),
            "magma",
            "Amplitude m$^{-1}$",
        ),
        (
            "semblance",
            surface_window_absmax(semblance, top, half_width_samples=half_width),
            "magma",
            "Discontinuity",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.8, 7.0), constrained_layout=True)
    x, y, margin = margin_contours(model)
    for axis, (name, values, cmap, colorbar_label), letter in zip(
        axes.ravel(), panels, "abcdef"
    ):
        values = values.T
        if name == "input":
            limit = float(np.percentile(np.abs(values), 99.0))
            image = axis.imshow(
                values,
                origin="lower",
                extent=(0, EXTENT_X_M, 0, EXTENT_Y_M),
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
        elif name == "footprint":
            image = axis.imshow(
                values,
                origin="lower",
                extent=(0, EXTENT_X_M, 0, EXTENT_Y_M),
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
        else:
            image = axis.imshow(
                values,
                origin="lower",
                extent=(0, EXTENT_X_M, 0, EXTENT_Y_M),
                cmap=cmap,
                vmin=0.0,
                vmax=float(np.percentile(values, 99.0)),
                interpolation="nearest",
            )
        if name != "footprint":
            axis.contour(
                x,
                y,
                margin,
                levels=[0.0],
                colors="white",
                linewidths=1.0,
                linestyles="--",
            )
            axis.contour(x, y, margin, levels=[0.0], colors="black", linewidths=0.35)
        if colorbar_label is not None:
            figure.colorbar(image, ax=axis, shrink=0.77, pad=0.02, label=colorbar_label)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        label_panel(axis, letter)
    figure.savefig(output / "channel_manuscript_comparison.png", dpi=600)
    plt.close(figure)


def summarize(
    invariance_rows: list[dict[str, object]],
    recovery_rows: list[dict[str, object]],
    contrast_rows: list[dict[str, object]],
) -> dict[str, object]:
    attributes = ("thdr", "asag", "tdr", "tahg")
    invariance: dict[str, dict[str, float]] = {}
    for attribute in attributes:
        invariance[attribute] = {}
        for mode in ("sample_space", "sampling_aware"):
            values = [
                float(row["normalized_rmse"])
                for row in invariance_rows
                if row["attribute"] == attribute and row["implementation"] == mode
            ]
            invariance[attribute][mode] = float(np.mean(values))
        before = invariance[attribute]["sample_space"]
        after = invariance[attribute]["sampling_aware"]
        invariance[attribute]["reduction_percent"] = 100.0 * (before - after) / before

    anisotropic = {"coarse_inline", "coarse_crossline"}
    recovery_attribute_names = {
        "thdr": "thdr",
        "asag": "asag",
        "tdr": "tdr_magnitude",
        "tahg": "tahg_magnitude",
    }
    orientation: dict[str, dict[str, float]] = {}
    for attribute in attributes:
        orientation[attribute] = {}
        recovery_name = recovery_attribute_names[attribute]
        for mode in ("sample_space", "sampling_aware"):
            values = [
                float(row["average_precision"])
                for row in recovery_rows
                if row["sampling_case"] in anisotropic
                and row["attribute"] == recovery_name
                and row["implementation"] == mode
            ]
            orientation[attribute][f"{mode}_mean_ap"] = float(np.mean(values))
            orientation[attribute][f"{mode}_sd_ap"] = float(np.std(values, ddof=1))

    clean_medium = [
        row
        for row in contrast_rows
        if float(row["contrast_fraction"]) == 0.06 and row["snr_db"] == "clean"
    ]
    clean_medium_ap = {
        f"{row['implementation']}:{row['attribute']}": float(row["average_precision"])
        for row in clean_medium
    }
    return {
        "invariance": invariance,
        "anisotropic_orientation_recovery": orientation,
        "clean_medium_contrast_average_precision": clean_medium_ap,
    }


def plot_metric_summary(output: Path, summary: dict[str, object]) -> None:
    attributes = ("thdr", "asag", "tdr", "tahg")
    invariance = summary["invariance"]
    orientation = summary["anisotropic_orientation_recovery"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    x = np.arange(len(attributes))
    width = 0.36
    colors = {"sample_space": "#c65d3a", "sampling_aware": "#2f7d8c"}
    for offset, mode in ((-width / 2, "sample_space"), (width / 2, "sampling_aware")):
        axes[0].bar(
            x + offset,
            [invariance[name][mode] for name in attributes],
            width,
            color=colors[mode],
            label=mode.replace("_", " "),
        )
        axes[1].bar(
            x + offset,
            [orientation[name][f"{mode}_mean_ap"] for name in attributes],
            width,
            yerr=[orientation[name][f"{mode}_sd_ap"] for name in attributes],
            capsize=3,
            color=colors[mode],
            label=mode.replace("_", " "),
        )
    axes[0].set_ylabel("Mean cross-grid normalized RMSE")
    axes[1].set_ylabel("Channel-margin average precision")
    for axis in axes:
        axis.set_xticks(x, [name.upper() for name in attributes])
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    for axis, letter in zip(axes, "ab"):
        label_panel(axis, letter)
    figure.savefig(output / "channel_metric_summary.png", dpi=240)
    plt.close(figure)


def plot_thdr_orientation_sensitivity(
    output: Path, recovery_rows: list[dict[str, object]]
) -> None:
    """Plot the direct channel test on the two interchanged anisotropic grids."""

    cases = ("coarse_inline", "coarse_crossline")
    colors = {"sample_space": "#c65d3a", "sampling_aware": "#2f7d8c"}
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True, constrained_layout=True)
    for axis, case, letter in zip(axes, cases, "ab"):
        for mode in ("sample_space", "sampling_aware"):
            selected = sorted(
                (
                    (float(row["orientation_deg"]), float(row["average_precision"]))
                    for row in recovery_rows
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
        axis.set_xticks(ORIENTATIONS_DEG)
        axis.set_ylim(0.60, 1.00)
        axis.spines[["top", "right"]].set_visible(False)
        label_panel(axis, letter)
    axes[0].set_ylabel("THDR channel-margin average precision")
    axes[1].legend(frameon=False, loc="lower right")
    figure.savefig(output / "thdr_orientation_sensitivity.png", dpi=240)
    plt.close(figure)


def write_report(output: Path, summary: dict[str, object]) -> None:
    invariance = summary["invariance"]
    orientation = summary["anisotropic_orientation_recovery"]
    ap = summary["clean_medium_contrast_average_precision"]
    lines = [
        "# Exploratory incised-channel benchmark",
        "",
        "## Purpose",
        "",
        "This experiment tests the numerical and interpretive behavior of sampling-aware gradient attributes on a planform channel target.",
        "",
        "## Design",
        "",
        "- Continuous incised meandering channel with semi-elliptical cross-section.",
        "- Smoothly varying impedance contrast along the channel.",
        "- Four independently generated grids and six channel orientations.",
        "- Channel-margin ground truth evaluated with average precision and best F1.",
        "- Low, medium, and high impedance contrasts, with clean and 10 dB cases.",
        "- Input amplitude and unsteered semblance-discontinuity baselines.",
        "",
        "## Cross-grid invariance",
        "",
        "| Attribute | Sample-space NRMSE | Sampling-aware NRMSE | Reduction |",
        "|---|---:|---:|---:|",
    ]
    for name in ("thdr", "asag", "tdr", "tahg"):
        values = invariance[name]
        lines.append(
            f"| {name.upper()} | {values['sample_space']:.3f} | {values['sampling_aware']:.3f} | {values['reduction_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Channel-margin recovery on anisotropic grids",
            "",
            "Values are means over six orientations and the two grids with interchanged 25 m and 12.5 m horizontal sampling. Standard deviations quantify orientation sensitivity.",
            "",
            "| Attribute | Sample-space AP | Sampling-aware AP | Sample-space SD | Sampling-aware SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("thdr", "asag", "tdr", "tahg"):
        values = orientation[name]
        lines.append(
            f"| {name.upper()} | {values['sample_space_mean_ap']:.3f} | {values['sampling_aware_mean_ap']:.3f} | {values['sample_space_sd_ap']:.3f} | {values['sampling_aware_sd_ap']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Medium-contrast clean reference case",
            "",
            "| Score | Average precision |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(ap.items()):
        lines.append(f"| {key} | {value:.3f} |")

    contrast_rows = []
    with (output / "channel_contrast_noise.csv").open(encoding="utf-8") as stream:
        contrast_rows = list(csv.DictReader(stream))
    lines.extend(
        [
            "",
            "## Noise stress test",
            "",
            "Average precision on the reference grid. The sampling-aware and sample-space THDR rankings coincide on this horizontally isotropic grid, so one THDR value is shown.",
            "",
            "| Contrast | Condition | Input amplitude | THDR | Semblance discontinuity |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for contrast in (0.03, 0.06, 0.12):
        for condition in ("clean", "10.0"):
            values = {
                row["attribute"]: float(row["average_precision"])
                for row in contrast_rows
                if float(row["contrast_fraction"]) == contrast
                and row["snr_db"] == condition
                and row["implementation"] in {"baseline", "sampling_aware"}
            }
            condition_label = "clean" if condition == "clean" else "10 dB"
            lines.append(
                f"| {100.0 * contrast:.0f}% | {condition_label} | {values['input_amplitude']:.3f} | {values['thdr']:.3f} | {values['semblance_discontinuity']:.3f} |"
            )

    thdr_gain = invariance["thdr"]["reduction_percent"]
    asag_gain = invariance["asag"]["reduction_percent"]
    thdr_ap_gain = orientation["thdr"]["sampling_aware_mean_ap"] - orientation["thdr"]["sample_space_mean_ap"]
    thdr_sd_gain = orientation["thdr"]["sample_space_sd_ap"] - orientation["thdr"]["sampling_aware_sd_ap"]
    strengthens_numeric = thdr_gain > 10.0 and asag_gain > 20.0
    strengthens_interpretive = thdr_ap_gain > 0.02 or thdr_sd_gain > 0.02
    lines.extend(
        [
            "",
            "## Scope of interpretation",
            "",
            f"- Numerical invariance criterion met: **{'yes' if strengthens_numeric else 'no'}**.",
            f"- THDR margin-recovery/orientation criterion met: **{'yes' if strengthens_interpretive else 'no'}**.",
            "- TDR zero-contour conditioning is reported separately and must not be confused with the raw TDR magnitude.",
            "- Physical scaling can reduce grid bias but cannot remove band limitation, tuning, or discretization error.",
            "- At 10 dB SNR, semblance outperforms THDR for the 3% and 6% contrasts; the channel result therefore supports complementary use, not universal superiority.",
            "",
            "## Benchmark conclusion",
            "",
            "The channel benchmark supports the sampling/azimuth test and gives THDR a clear interpretive role. It does not make all four raw attributes equally useful: ASAG is stable but not margin-specific, raw TDR magnitude is not an edge score, and TAHG remains oscillatory. The channel and fault experiments are therefore complementary demonstrations rather than evidence that one attribute is universally superior.",
            "",
        ]
    )
    (output / "CHANNEL_BENCHMARK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output = ROOT / "output" / "channel_benchmark_v0.1"
    output.mkdir(parents=True, exist_ok=True)
    if "--figures-only" in sys.argv[1:]:
        print("Rendering the manuscript channel figure...")
        plot_manuscript_channel_comparison(output)
        print(f"Figure written to {output}")
        return
    print("Running channel grid and orientation benchmark...")
    invariance_rows, recovery_rows = invariance_and_orientation_experiment(output)
    print("Running channel contrast/noise benchmark...")
    contrast_rows = contrast_noise_experiment(output)
    print("Rendering channel figures...")
    plot_channel_model(output)
    plot_attribute_maps(output)
    plot_manuscript_channel_comparison(output)
    summary = summarize(invariance_rows, recovery_rows, contrast_rows)
    plot_metric_summary(output, summary)
    plot_thdr_orientation_sensitivity(output, recovery_rows)
    (output / "channel_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(output, summary)
    print(json.dumps(summary, indent=2))
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
