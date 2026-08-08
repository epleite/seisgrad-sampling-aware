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
