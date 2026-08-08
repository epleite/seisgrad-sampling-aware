"""Simple normal-incidence post-stack forward modelling."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d


def acoustic_impedance(vp: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Return P-wave acoustic impedance."""

    return np.asarray(vp, dtype=float) * np.asarray(density, dtype=float)


def ricker_wavelet(frequency_hz: float, dt_s: float, duration_s: float = 0.128) -> np.ndarray:
    """Return a zero-phase Ricker wavelet sampled at ``dt_s``."""

    half = duration_s / 2.0
    t = np.arange(-half, half + 0.5 * dt_s, dt_s)
    a = (np.pi * frequency_hz * t) ** 2
    wavelet = (1.0 - 2.0 * a) * np.exp(-a)
    return wavelet / np.max(np.abs(wavelet))


def _reflectivity_in_regular_time(
    vp: np.ndarray,
    density: np.ndarray,
    dz_m: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    impedance = acoustic_impedance(vp, density)
    reflectivity_depth = np.zeros_like(impedance, dtype=float)
    numerator = impedance[..., 1:] - impedance[..., :-1]
    denominator = impedance[..., 1:] + impedance[..., :-1]
    reflectivity_depth[..., 1:] = numerator / np.maximum(denominator, 1e-12)

    layer_dt = 2.0 * dz_m / np.maximum(vp, 1.0)
    twt = np.cumsum(layer_dt, axis=-1)
    twt -= twt[..., :1]
    t_max = float(np.max(twt))
    time_s = np.arange(0.0, t_max + 0.5 * dt_s, dt_s)
    nt = time_s.size

    ntraces = vp.shape[0] * vp.shape[1]
    twt_flat = twt.reshape(ntraces, vp.shape[-1])
    refl_flat = reflectivity_depth.reshape(ntraces, vp.shape[-1])
    index_float = twt_flat / dt_s
    index0 = np.floor(index_float).astype(np.int64)
    fraction = index_float - index0

    output = np.zeros(ntraces * nt, dtype=float)
    trace_offset = (np.arange(ntraces, dtype=np.int64) * nt)[:, None]

    valid0 = index0 < nt
    np.add.at(
        output,
        (trace_offset + np.minimum(index0, nt - 1))[valid0],
        (refl_flat * (1.0 - fraction))[valid0],
    )
    valid1 = index0 + 1 < nt
    np.add.at(
        output,
        (trace_offset + np.minimum(index0 + 1, nt - 1))[valid1],
        (refl_flat * fraction)[valid1],
    )

    return output.reshape(vp.shape[0], vp.shape[1], nt), time_s


def poststack_synthetic(
    vp: np.ndarray,
    density: np.ndarray,
    *,
    dz_m: float,
    dt_s: float = 0.002,
    frequency_hz: float = 30.0,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a band-limited normal-incidence post-stack volume."""

    reflectivity, time_s = _reflectivity_in_regular_time(vp, density, dz_m, dt_s)
    wavelet = ricker_wavelet(frequency_hz, dt_s)
    seismic = convolve1d(reflectivity, wavelet, axis=-1, mode="constant", cval=0.0)
    if normalize:
        scale = np.percentile(np.abs(seismic), 99.5)
        if scale > 0.0:
            seismic = seismic / scale
    return seismic.astype(np.float32), time_s


def depth_property_to_regular_time(
    property_depth: np.ndarray,
    vp: np.ndarray,
    *,
    dz_m: float,
    time_s: np.ndarray,
) -> np.ndarray:
    """Interpolate a depth-domain property onto the regular two-way-time grid."""

    property_depth = np.asarray(property_depth, dtype=float)
    vp = np.asarray(vp, dtype=float)
    if property_depth.shape != vp.shape or property_depth.ndim != 3:
        raise ValueError("property_depth and vp must be matching 3D arrays")

    layer_dt = 2.0 * dz_m / np.maximum(vp, 1.0)
    twt = np.cumsum(layer_dt, axis=-1)
    twt -= twt[..., :1]
    ntraces = vp.shape[0] * vp.shape[1]
    twt_flat = twt.reshape(ntraces, vp.shape[-1])
    property_flat = property_depth.reshape(ntraces, vp.shape[-1])
    output = np.empty((ntraces, time_s.size), dtype=np.float32)
    for trace in range(ntraces):
        output[trace] = np.interp(
            time_s,
            twt_flat[trace],
            property_flat[trace],
            left=property_flat[trace, 0],
            right=property_flat[trace, -1],
        )
    return output.reshape(vp.shape[0], vp.shape[1], time_s.size)


def depth_mask_to_regular_time(
    mask_depth: np.ndarray,
    vp: np.ndarray,
    *,
    dz_m: float,
    time_s: np.ndarray,
    half_width_samples: int = 1,
) -> np.ndarray:
    """Map a depth-domain Boolean surface mask to regular two-way time."""

    mask_depth = np.asarray(mask_depth, dtype=bool)
    vp = np.asarray(vp, dtype=float)
    if mask_depth.shape != vp.shape or mask_depth.ndim != 3:
        raise ValueError("mask_depth and vp must be matching 3D arrays")
    if half_width_samples < 0:
        raise ValueError("half_width_samples must be non-negative")

    dt_s = float(np.median(np.diff(time_s)))
    layer_dt = 2.0 * dz_m / np.maximum(vp, 1.0)
    twt = np.cumsum(layer_dt, axis=-1)
    twt -= twt[..., :1]
    output = np.zeros((vp.shape[0], vp.shape[1], time_s.size), dtype=bool)
    coordinates = np.argwhere(mask_depth)
    for ix, iy, iz in coordinates:
        center = int(round(twt[ix, iy, iz] / dt_s))
        start = max(0, center - half_width_samples)
        stop = min(time_s.size, center + half_width_samples + 1)
        output[ix, iy, start:stop] = True
    return output
