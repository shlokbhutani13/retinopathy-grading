import numpy as np
import pytest

from retinopathy.analysis import clustered_bootstrap_confidence_interval


def test_clustered_bootstrap_is_deterministic_and_keeps_patients_intact():
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    predictions = np.array([0, 0, 1, 2, 2, 2, 4, 3])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    observed_shapes = []

    def accuracy(truth, predicted):
        assert len(truth) % 2 == 0
        assert np.all(truth.reshape(-1, 2)[:, 0] == truth.reshape(-1, 2)[:, 1])
        observed_shapes.append(len(truth))
        return float(np.mean(truth == predicted))

    first = clustered_bootstrap_confidence_interval(
        labels,
        predictions,
        groups,
        metric=accuracy,
        samples=50,
        seed=7,
    )
    second = clustered_bootstrap_confidence_interval(
        labels,
        predictions,
        groups,
        metric=accuracy,
        samples=50,
        seed=7,
    )

    assert first == second
    assert first["lower"] <= first["estimate"] <= first["upper"]
    assert set(observed_shapes) == {8}


def test_clustered_bootstrap_rejects_mismatched_group_count():
    with pytest.raises(ValueError, match="same number of rows"):
        clustered_bootstrap_confidence_interval(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array(["a"]),
            metric=lambda truth, predicted: float(np.mean(truth == predicted)),
        )
