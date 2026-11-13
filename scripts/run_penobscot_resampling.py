"""Reproduce the public Penobscot field-resampling experiment.

The input is ``Penobscot/seismic.npy`` from the AdaSemSeg processed-data
archive (Zenodo DOI 10.5281/zenodo.21764042).  The external data are not
redistributed here.  Labels are neither loaded nor used.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import firwin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seisgrad.attributes import gradient_attributes
from seisgrad.metrics import rms_normalized_error


NATIVE_SPACING = (12.5, 25.01, 0.004)
SUBVOLUME_SHAPE = (192, 192, 384)
VELOCITIES_M_S = (2550.0, 3000.0, 3450.0)
ATTRIBUTES = ("thdr", "asag", "tdr", "tahg")
OUTPUT = ROOT / "output" / "revision_v1.1.3"


def central_subvolume(volume: np.ndarray) -> np.ndarray:
    """Return the documented central 192 x 192 x 384 amplitude block."""

    if volume.ndim != 3:
        raise ValueError("Penobscot amplitude must be a 3D array")
    starts = tuple((size - wanted) // 2 for size, wanted in zip(volume.shape, SUBVOLUME_SHAPE))
    if any(start < 0 for start in starts):
        raise ValueError(
            f"input shape {volume.shape} is smaller than {SUBVOLUME_SHAPE}"
        )
    slices = tuple(
        slice(start, start + wanted)
        for start, wanted in zip(starts, SUBVOLUME_SHAPE)
    )
    block = np.asarray(volume[slices], dtype=np.float32)
    block -= np.median(block)
    scale = float(np.percentile(np.abs(block), 99.0))
    if scale <= 0.0:
        raise ValueError("the selected subvolume has zero amplitude scale")
    return block / scale


def anti_alias_filter(
    amplitude: np.ndarray,
    *,
    axis: int,
    beta: float,
) -> np.ndarray:
    """Apply the documented 31-tap Kaiser FIR before factor-two decimation."""

    taps = firwin(31, 0.45, window=("kaiser", beta))
    return convolve1d(amplitude, taps, axis=axis, mode="reflect")


def crop_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude 20 horizontal and eight temporal samples on the target grid."""

    interior = (slice(20, -20), slice(20, -20), slice(8, -8))
    return reference[interior], estimate[interior]


def decimate(amplitude: np.ndarray, axis: int) -> np.ndarray:
    slices = [slice(None)] * amplitude.ndim
    slices[axis] = slice(None, None, 2)
    return amplitude[tuple(slices)]


def attribute_set(
    amplitude: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    velocity_m_s: float,
    sampling_aware: bool,
) -> dict[str, np.ndarray]:
    return gradient_attributes(
        amplitude,
        spacing=spacing,
        reference_velocity_m_s=velocity_m_s,
        sampling_aware=sampling_aware,
    )


def evaluate_axis(
    amplitude: np.ndarray,
    *,
    axis: int,
    beta: float,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    """Filter, decimate, and compare the two derivative implementations."""

    filtered = anti_alias_filter(amplitude, axis=axis, beta=beta)
    target = decimate(filtered, axis)
    target_spacing = list(NATIVE_SPACING)
    target_spacing[axis] *= 2.0
    target_spacing_tuple = tuple(target_spacing)

    native_references = {
        implementation: attribute_set(
            filtered,
            spacing=NATIVE_SPACING,
            velocity_m_s=3000.0,
            sampling_aware=aware,
        )
        for implementation, aware in (
            ("sample_space", False),
            ("sampling_aware", True),
        )
    }
    references = {
        implementation: {
            name: decimate(values, axis)
            for name, values in attribute_values.items()
        }
        for implementation, attribute_values in native_references.items()
    }
    del native_references

    rows: list[dict[str, object]] = []
    panels: dict[str, np.ndarray] = {"amplitude": target}
    for velocity in VELOCITIES_M_S:
        for implementation, aware in (
            ("sample_space", False),
            ("sampling_aware", True),
        ):
            attributes = attribute_set(
                target,
                spacing=target_spacing_tuple,
                velocity_m_s=velocity,
                sampling_aware=aware,
            )
            for name in ATTRIBUTES:
                reference_modes = (
                    ("shared_sampling_aware_native", "sampling_aware"),
                    ("implementation_matched_native", implementation),
                )
                for reference_definition, reference_mode in reference_modes:
                    ref_crop, estimate_crop = crop_pair(
                        references[reference_mode][name], attributes[name]
                    )
                    rows.append(
                        {
                            "preprocessing": (
                                f"31_tap_kaiser_beta_{beta:g}_cutoff_0.45"
                            ),
                            "decimated_axis": (
                                "inline" if axis == 0 else "crossline"
                            ),
                            "target_dx_m": target_spacing_tuple[0],
                            "target_dy_m": target_spacing_tuple[1],
                            "dt_s": target_spacing_tuple[2],
                            "velocity_m_s": velocity,
                            "reference_definition": reference_definition,
                            "implementation": implementation,
                            "attribute": name,
                            "rms_normalized_error": rms_normalized_error(
                                ref_crop, estimate_crop
                            ),
                        }
                    )
            if velocity == 3000.0:
                for name in ("thdr", "tdr"):
                    panels[f"{implementation}_{name}"] = attributes[name]

    for name in ("thdr", "tdr"):
        panels[f"reference_{name}"] = references["sampling_aware"][name]
    return rows, panels


def evaluate_unfiltered_control(
    amplitude: np.ndarray,
    *,
    axis: int,
) -> list[dict[str, object]]:
    """Quantify direct decimation without the anti-alias filter at 3000 m/s."""

    target = decimate(amplitude, axis)
    target_spacing = list(NATIVE_SPACING)
    target_spacing[axis] *= 2.0
    target_spacing_tuple = tuple(target_spacing)
    native_references = {
        implementation: attribute_set(
            amplitude,
            spacing=NATIVE_SPACING,
            velocity_m_s=3000.0,
            sampling_aware=aware,
        )
        for implementation, aware in (
            ("sample_space", False),
            ("sampling_aware", True),
        )
    }
    references = {
        implementation: {
            name: decimate(values, axis)
            for name, values in attribute_values.items()
        }
        for implementation, attribute_values in native_references.items()
    }
    rows: list[dict[str, object]] = []
    for implementation, aware in (
        ("sample_space", False),
        ("sampling_aware", True),
    ):
        attributes = attribute_set(
            target,
            spacing=target_spacing_tuple,
            velocity_m_s=3000.0,
            sampling_aware=aware,
        )
        for name in ATTRIBUTES:
            reference_modes = (
                ("shared_sampling_aware_native", "sampling_aware"),
                ("implementation_matched_native", implementation),
            )
            for reference_definition, reference_mode in reference_modes:
                ref_crop, estimate_crop = crop_pair(
                    references[reference_mode][name], attributes[name]
                )
                rows.append(
                    {
                        "preprocessing": "unfiltered_aliasing_control",
                        "decimated_axis": (
                            "inline" if axis == 0 else "crossline"
                        ),
                        "target_dx_m": target_spacing_tuple[0],
                        "target_dy_m": target_spacing_tuple[1],
                        "dt_s": target_spacing_tuple[2],
                        "velocity_m_s": 3000.0,
                        "reference_definition": reference_definition,
                        "implementation": implementation,
                        "attribute": name,
                        "rms_normalized_error": rms_normalized_error(
                            ref_crop, estimate_crop
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_crossline_panels(panels: dict[str, np.ndarray]) -> None:
    """Write the qualitative before/after time-slice comparison."""

    time_index = panels["amplitude"].shape[2] // 2
    keys = (
        "reference_thdr",
        "sample_space_thdr",
        "sampling_aware_thdr",
        "reference_tdr",
        "sample_space_tdr",
        "sampling_aware_tdr",
    )
    column_titles = ("Native reference", "Conventional", "Sampling-aware")
    dx_km = 12.5 / 1000.0
    dy_km = 50.02 / 1000.0
    nx, ny = panels["amplitude"].shape[:2]
    extent = (0.0, (nx - 1) * dx_km, 0.0, (ny - 1) * dy_km)
    x_coordinates = np.arange(nx) * dx_km
    y_coordinates = np.arange(ny) * dy_km

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9.3, 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    thdr_mappable = None
    tdr_mappable = None
    for panel_index, (axis, key) in enumerate(zip(axes.ravel(), keys)):
        image = panels[key][:, :, time_index].T
        if key.endswith("thdr"):
            limit = float(np.percentile(np.abs(image), 99.0))
            thdr_mappable = axis.imshow(
                image / max(limit, 1e-12),
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
            )
        else:
            tdr_mappable = axis.imshow(
                image,
                cmap="coolwarm",
                vmin=-np.pi / 2.0,
                vmax=np.pi / 2.0,
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
            )
            axis.contour(
                x_coordinates,
                y_coordinates,
                image,
                levels=(0.0,),
                colors="0.15",
                linewidths=0.25,
                alpha=0.55,
            )
        axis.text(
            0.025,
            0.965,
            f"({chr(97 + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.3},
        )
        axis.tick_params(labelsize=7.5, direction="out", length=2.5)
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=9.5, pad=6)
    for axis in axes[1]:
        axis.set_xlabel("Inline distance (km)", fontsize=8.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Crossline distance (km)", fontsize=8.5)
    axes[0, 0].annotate(
        "THDR",
        xy=(-0.31, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )
    axes[1, 0].annotate(
        "TDR",
        xy=(-0.31, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )
    if thdr_mappable is None or tdr_mappable is None:
        raise RuntimeError("missing THDR or TDR panels")
    thdr_bar = figure.colorbar(
        thdr_mappable,
        ax=axes[0, :],
        location="right",
        fraction=0.035,
        pad=0.015,
    )
    thdr_bar.set_label("THDR / P99(THDR)", fontsize=8.5)
    thdr_bar.ax.tick_params(labelsize=7.5)
    tdr_bar = figure.colorbar(
        tdr_mappable,
        ax=axes[1, :],
        location="right",
        fraction=0.035,
        pad=0.015,
    )
    tdr_bar.set_label("TDR (rad)", fontsize=8.5)
    tdr_bar.set_ticks((-np.pi / 2.0, 0.0, np.pi / 2.0))
    tdr_bar.set_ticklabels((r"$-\pi/2$", "0", r"$\pi/2$"))
    tdr_bar.ax.tick_params(labelsize=7.5)
    figure.savefig(
        OUTPUT / "figure_5_penobscot_time_slice.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        type=Path,
        help="Path to Penobscot/seismic.npy from Zenodo record 21764042.",
    )
    parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=8.0,
        help="Kaiser-window beta (default: 8.0, as used in the manuscript).",
    )
    args = parser.parse_args()

    volume = np.load(args.data, mmap_mode="r")
    amplitude = central_subvolume(volume)
    all_rows: list[dict[str, object]] = []
    for axis in (0, 1):
        print(
            f"Running Penobscot {'inline' if axis == 0 else 'crossline'} decimation...",
            flush=True,
        )
        rows, panels = evaluate_axis(amplitude, axis=axis, beta=args.kaiser_beta)
        all_rows.extend(rows)
        all_rows.extend(evaluate_unfiltered_control(amplitude, axis=axis))
        if axis == 1:
            plot_crossline_panels(panels)
    write_csv(OUTPUT / "penobscot_resampling.csv", all_rows)
    print(f"Penobscot results written to {OUTPUT}")


if __name__ == "__main__":
    main()
