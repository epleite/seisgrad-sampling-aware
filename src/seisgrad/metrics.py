"""Compact invariance metrics used by the prototype."""

from __future__ import annotations

import numpy as np


def normalized_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("arrays must have identical shapes")
    scale = np.sqrt(np.mean(reference**2))
    return float(np.sqrt(np.mean((reference - estimate) ** 2)) / max(scale, 1e-12))


def shape_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """RMSE after independently scaling both arrays to unit RMS.

    This separates spatial-pattern disagreement from a global positive scale
    factor.  It is therefore the primary cross-grid metric in the manuscript
    revision, while :func:`normalized_rmse` remains amplitude sensitive.
    """

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("arrays must have identical shapes")
    reference_scale = max(float(np.sqrt(np.mean(reference**2))), 1e-12)
    estimate_scale = max(float(np.sqrt(np.mean(estimate**2))), 1e-12)
    return float(
        np.sqrt(
            np.mean(
                (reference / reference_scale - estimate / estimate_scale) ** 2
            )
        )
    )


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    if reference.size != estimate.size:
        raise ValueError("arrays must contain the same number of values")
    if np.std(reference) == 0.0 or np.std(estimate) == 0.0:
        return float("nan")
    return float(np.corrcoef(reference, estimate)[0, 1])


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision for a strongly imbalanced binary target."""

    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.size != scores.size:
        raise ValueError("labels and scores must contain the same number of values")
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")[::-1]
    ranked = labels[order]
    true_positives = np.cumsum(ranked)
    precision = true_positives / np.arange(1, ranked.size + 1)
    return float(np.sum(precision[ranked]) / positives)


def best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return maximum F1 and the corresponding score threshold."""

    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.size != scores.size:
        raise ValueError("labels and scores must contain the same number of values")
    positives = int(labels.sum())
    if positives == 0:
        return float("nan"), float("nan")
    order = np.argsort(scores, kind="stable")[::-1]
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    true_positives = np.cumsum(ranked_labels)
    predicted = np.arange(1, ranked_labels.size + 1)
    precision = true_positives / predicted
    recall = true_positives / positives
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1))
    return float(f1[index]), float(ranked_scores[index])


def add_gaussian_noise(
    data: np.ndarray,
    snr_db: float | None,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add zero-mean Gaussian noise at a specified RMS signal-to-noise ratio."""

    data = np.asarray(data, dtype=float)
    if snr_db is None or np.isinf(snr_db):
        return data.copy()
    signal_rms = float(np.sqrt(np.mean(data**2)))
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    return data + rng.normal(0.0, noise_rms, size=data.shape)


def add_colored_impulsive_noise(
    data: np.ndarray,
    snr_db: float,
    *,
    rng: np.random.Generator,
    footprint_period_samples: float = 12.0,
) -> np.ndarray:
    """Add colored, heavy-tailed noise and a periodic acquisition footprint.

    A shared realization is generated once and must be passed unchanged to the
    two derivative implementations in a paired experiment.  The composite is
    normalized to the requested RMS SNR after its three components are added.
    """

    from scipy.ndimage import gaussian_filter

    values = np.asarray(data, dtype=float)
    colored = gaussian_filter(
        rng.normal(size=values.shape), sigma=(1.6, 0.8, 0.0), mode="reflect"
    )
    colored /= max(float(np.sqrt(np.mean(colored**2))), 1e-12)

    impulsive = rng.standard_t(df=3.0, size=values.shape)
    impulsive = np.clip(impulsive, -12.0, 12.0)
    impulsive /= max(float(np.sqrt(np.mean(impulsive**2))), 1e-12)

    phase = rng.uniform(0.0, 2.0 * np.pi)
    footprint_1d = np.sin(
        2.0 * np.pi * np.arange(values.shape[0]) / footprint_period_samples + phase
    )
    footprint = np.broadcast_to(footprint_1d[:, None, None], values.shape)

    composite = 0.65 * colored + 0.25 * impulsive + 0.35 * footprint
    composite -= float(np.mean(composite))
    composite /= max(float(np.sqrt(np.mean(composite**2))), 1e-12)
    signal_rms = float(np.sqrt(np.mean(values**2)))
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    return values + noise_rms * composite


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a paired mean change."""

    values = np.asarray(differences, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("at least two paired differences are required")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return float(low), float(high)
