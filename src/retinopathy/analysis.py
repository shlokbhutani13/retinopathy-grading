from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def portable_identifiers(paths: Sequence[str]) -> list[str]:
    return [Path(path).name for path in paths]


def bootstrap_confidence_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels and predictions must contain the same number of rows")
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = generator.integers(0, len(labels), len(labels))
        values.append(metric(labels[indices], predictions[indices]))
    tail = (1.0 - confidence) / 2
    return {
        "estimate": float(metric(labels, predictions)),
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
    }


def clustered_bootstrap_confidence_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    groups = np.asarray(groups)
    if not (len(labels) == len(predictions) == len(groups)):
        raise ValueError(
            "labels, predictions, and groups must contain the same number of rows"
        )
    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled_groups = generator.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        values.append(metric(labels[indices], predictions[indices]))
    tail = (1.0 - confidence) / 2
    return {
        "estimate": float(metric(labels, predictions)),
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
    }


def failure_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    paths: Sequence[str],
    confidence_threshold: float = 0.75,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    mask = (predictions != labels) & (confidence >= confidence_threshold)
    table = pd.DataFrame(
        {
            "path": np.asarray(paths)[mask],
            "true_grade": labels[mask],
            "predicted_grade": predictions[mask],
            "confidence": confidence[mask],
        }
    )
    return table.sort_values("confidence", ascending=False).reset_index(drop=True)
