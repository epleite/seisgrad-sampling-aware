"""Conventional and sampling-aware gradient-derived seismic attributes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GradientComponents:
    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray


def _derivative(
    data: np.ndarray,
    *,
    spacing: float,
    axis: int,
    order: int,
) -> np.ndarray:
    """Differentiate along one axis with documented boundary treatment.

    The order-2 implementation is NumPy's centered three-point operator with
    second-order one-sided boundary values.  The order-4 implementation
    replaces only the interior with the centered five-point formula; the two
    samples nearest either boundary retain the order-2 result.
    """

    if order not in (2, 4):
        raise ValueError("derivative_order must be 2 or 4")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    values = np.asarray(data, dtype=float)
    derivative = np.gradient(values, spacing, axis=axis, edge_order=2)
    if order == 2 or values.shape[axis] < 5:
        return derivative

    center = [slice(None)] * values.ndim
    minus_two = center.copy()
    minus_one = center.copy()
    plus_one = center.copy()
    plus_two = center.copy()
    center[axis] = slice(2, -2)
    minus_two[axis] = slice(0, -4)
    minus_one[axis] = slice(1, -3)
    plus_one[axis] = slice(3, -1)
    plus_two[axis] = slice(4, None)
    derivative[tuple(center)] = (
        values[tuple(minus_two)]
        - 8.0 * values[tuple(minus_one)]
        + 8.0 * values[tuple(plus_one)]
        - values[tuple(plus_two)]
    ) / (12.0 * spacing)
    return derivative


def _components(
    amplitude: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    reference_velocity_m_s: float | np.ndarray,
    sampling_aware: bool,
    derivative_order: int,
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
        gx = _derivative(data, spacing=dx_m, axis=0, order=derivative_order)
        gy = _derivative(data, spacing=dy_m, axis=1, order=derivative_order)
        gt = _derivative(data, spacing=dt_s, axis=2, order=derivative_order)
        gz = (2.0 / velocity) * gt
    else:
        gx = _derivative(data, spacing=1.0, axis=0, order=derivative_order)
        gy = _derivative(data, spacing=1.0, axis=1, order=derivative_order)
        gz = _derivative(data, spacing=1.0, axis=2, order=derivative_order)

    return GradientComponents(gx=gx, gy=gy, gz=gz)


def gradient_attributes(
    amplitude: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    reference_velocity_m_s: float | np.ndarray = 3000.0,
    sampling_aware: bool = True,
    derivative_order: int = 2,
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
    derivative_order:
        Centered finite-difference order.  Order 4 uses a five-point interior
        stencil and retains the order-2 treatment for the two samples nearest
        each boundary.
    """

    components = _components(
        amplitude,
        spacing=spacing,
        reference_velocity_m_s=reference_velocity_m_s,
        sampling_aware=sampling_aware,
        derivative_order=derivative_order,
    )
    gx, gy, gz = components.gx, components.gy, components.gz
    thdr = np.hypot(gx, gy)
    asag = np.sqrt(thdr**2 + gz**2)
    tdr = np.arctan2(gz, thdr)

    if sampling_aware:
        dx_m, dy_m, dt_s = spacing
        hx = _derivative(thdr, spacing=dx_m, axis=0, order=derivative_order)
        hy = _derivative(thdr, spacing=dy_m, axis=1, order=derivative_order)
        ht = _derivative(thdr, spacing=dt_s, axis=2, order=derivative_order)
        velocity = np.asarray(reference_velocity_m_s, dtype=float)
        hz = (2.0 / velocity) * ht
    else:
        hx = _derivative(thdr, spacing=1.0, axis=0, order=derivative_order)
        hy = _derivative(thdr, spacing=1.0, axis=1, order=derivative_order)
        hz = _derivative(thdr, spacing=1.0, axis=2, order=derivative_order)
    tahg = np.arctan2(hz, np.hypot(hx, hy))

    return {
        "thdr": thdr.astype(np.float32),
        "asag": asag.astype(np.float32),
        "tdr": tdr.astype(np.float32),
        "tahg": tahg.astype(np.float32),
    }
