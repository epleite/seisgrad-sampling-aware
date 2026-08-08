"""Conventional and sampling-aware gradient-derived seismic attributes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GradientComponents:
    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray


def _components(
    amplitude: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    reference_velocity_m_s: float | np.ndarray,
    sampling_aware: bool,
) -> GradientComponents:
    data = np.asarray(amplitude, dtype=float)
    if data.ndim != 3:
        raise ValueError("amplitude must be a 3D array ordered as (x, y, time)")

    if sampling_aware:
        dx_m, dy_m, dt_s = spacing
        velocity = np.asarray(reference_velocity_m_s, dtype=float)
        if min(dx_m, dy_m, dt_s) <= 0.0 or np.any(velocity <= 0.0):
            raise ValueError("spacing and reference velocity must be positive")
        if velocity.ndim > 0 and velocity.shape != data.shape:
            raise ValueError("a velocity field must match the amplitude volume")
        gx, gy, gt = np.gradient(data, dx_m, dy_m, dt_s, edge_order=2)
        gz = (2.0 / velocity) * gt
    else:
        gx, gy, gz = np.gradient(data, edge_order=2)

    return GradientComponents(gx=gx, gy=gy, gz=gz)


def gradient_attributes(
    amplitude: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    reference_velocity_m_s: float | np.ndarray = 3000.0,
    sampling_aware: bool = True,
) -> dict[str, np.ndarray]:
    """Calculate THDR, ASAG, TDR, and TAHG from a seismic volume.

    Parameters
    ----------
    amplitude:
        Seismic volume ordered as `(x, y, time)`.
    spacing:
        `(dx_m, dy_m, dt_s)` for the physical grid.
    reference_velocity_m_s:
        Scalar velocity or a velocity field used to convert the temporal
        derivative to a pseudo-depth derivative. It is ignored by the
        conventional sample-space version.
    sampling_aware:
        If false, all derivatives are taken per sample, reproducing the common
        index-space implementation.
    """

    components = _components(
        amplitude,
        spacing=spacing,
        reference_velocity_m_s=reference_velocity_m_s,
        sampling_aware=sampling_aware,
    )
    gx, gy, gz = components.gx, components.gy, components.gz
    thdr = np.hypot(gx, gy)
    asag = np.sqrt(thdr**2 + gz**2)
    tdr = np.arctan2(gz, thdr)

    if sampling_aware:
        dx_m, dy_m, dt_s = spacing
        hx, hy, ht = np.gradient(thdr, dx_m, dy_m, dt_s, edge_order=2)
        velocity = np.asarray(reference_velocity_m_s, dtype=float)
        hz = (2.0 / velocity) * ht
    else:
        hx, hy, hz = np.gradient(thdr, edge_order=2)
    tahg = np.arctan2(hz, np.hypot(hx, hy))

    return {
        "thdr": thdr.astype(np.float32),
        "asag": asag.astype(np.float32),
        "tdr": tdr.astype(np.float32),
        "tahg": tahg.astype(np.float32),
    }
