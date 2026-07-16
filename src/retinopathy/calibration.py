from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits and labels have incompatible shapes")

    def negative_log_likelihood(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = softmax(logits / temperature)
        selected = probabilities[np.arange(len(labels)), labels]
        return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(-3.0, 3.0),
        method="bounded",
    )
    temperature = float(np.exp(result.x))
    return temperature, softmax(logits / temperature)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        accuracy_gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        error += float(mask.mean()) * accuracy_gap
    return float(error)
