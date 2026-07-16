import numpy as np

from retinopathy.calibration import expected_calibration_error, temperature_scale
from retinopathy.evaluation import evaluate_predictions


def test_evaluation_reports_ordered_and_screening_metrics():
    labels = np.array([0, 1, 2, 3, 4, 0, 2, 4])
    probabilities = np.eye(5)[labels] * 0.8 + 0.04
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    metrics = evaluate_predictions(labels, probabilities)

    assert metrics["quadratic_weighted_kappa"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["referable_sensitivity"] == 1.0
    assert metrics["referable_specificity"] == 1.0
    assert np.asarray(metrics["confusion_matrix"]).shape == (5, 5)


def test_temperature_scaling_returns_probabilities():
    logits = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    labels = np.array([0, 1])

    temperature, probabilities = temperature_scale(logits, labels)

    assert temperature > 0
    assert probabilities.shape == logits.shape
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_expected_calibration_error_is_bounded():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.55, 0.45]])
    labels = np.array([0, 1, 1])

    value = expected_calibration_error(probabilities, labels, bins=3)

    assert 0 <= value <= 1
