"""Compact reference attributes for the synthetic benchmark."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def unsteered_semblance_discontinuity(
    amplitude: np.ndarray,
    *,
    spatial_window: tuple[int, int] = (3, 3),
    temporal_window: int = 9,
) -> np.ndarray:
    """Return one minus local unsteered semblance.

    This is intentionally a simple, transparent baseline rather than a
    production dip-steered coherence implementation.
    """

    data = np.asarray(amplitude, dtype=float)
    wx, wy = spatial_window
    if data.ndim != 3 or min(wx, wy, temporal_window) < 1:
        raise ValueError("invalid volume or window")
    ntraces = wx * wy
    spatial_sum = uniform_filter(data, size=(wx, wy, 1), mode="nearest") * ntraces
    spatial_energy = uniform_filter(data**2, size=(wx, wy, 1), mode="nearest") * ntraces
    numerator = uniform_filter(spatial_sum**2, size=(1, 1, temporal_window), mode="nearest")
    denominator = ntraces * uniform_filter(
        spatial_energy, size=(1, 1, temporal_window), mode="nearest"
    )
    semblance = numerator / np.maximum(denominator, 1e-12)
    return np.clip(1.0 - semblance, 0.0, 1.0).astype(np.float32)
