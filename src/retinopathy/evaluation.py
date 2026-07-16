from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("probabilities must have shape (samples, 5)")
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("labels and probabilities must contain the same number of samples")

    predictions = probabilities.argmax(axis=1)
    referable_labels = (labels >= 2).astype(int)
    referable_probabilities = probabilities[:, 2:].sum(axis=1)
    referable_predictions = (referable_probabilities >= 0.5).astype(int)
    specificity = recall_score(
        referable_labels,
        referable_predictions,
        pos_label=0,
        zero_division=0,
    )
    negative_predictive_value = precision_score(
        referable_labels,
        referable_predictions,
        pos_label=0,
        zero_division=0,
    )

    return {
        "quadratic_weighted_kappa": float(
            _quadratic_weighted_kappa(labels, predictions, classes=5)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "per_class_recall": recall_score(
            labels,
            predictions,
            labels=np.arange(5),
            average=None,
            zero_division=0,
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=np.arange(5),
        ).tolist(),
        "referable_auroc": float(roc_auc_score(referable_labels, referable_probabilities)),
        "referable_sensitivity": float(
            recall_score(referable_labels, referable_predictions, zero_division=0)
        ),
        "referable_specificity": float(specificity),
        "referable_precision": float(
            precision_score(referable_labels, referable_predictions, zero_division=0)
        ),
        "referable_negative_predictive_value": float(negative_predictive_value),
    }


def _quadratic_weighted_kappa(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    classes: int,
) -> float:
    observed = confusion_matrix(labels, predictions, labels=np.arange(classes)).astype(float)
    total = observed.sum()
    if total == 0:
        return 0.0
    actual = observed.sum(axis=1)
    predicted = observed.sum(axis=0)
    expected = np.outer(actual, predicted) / total
    indices = np.arange(classes)
    weights = (indices[:, None] - indices[None, :]) ** 2 / (classes - 1) ** 2
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 1.0
    return 1.0 - float((weights * observed).sum()) / denominator
