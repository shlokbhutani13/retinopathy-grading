import numpy as np

from retinopathy.analysis import (
    bootstrap_confidence_interval,
    failure_table,
    portable_identifiers,
)


def test_bootstrap_interval_is_deterministic_and_contains_estimate():
    labels = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    predictions = np.array([0, 0, 1, 0, 1, 0, 1, 0])

    result = bootstrap_confidence_interval(
        labels,
        predictions,
        metric=lambda truth, pred: float((truth == pred).mean()),
        samples=200,
        seed=17,
    )

    assert result == bootstrap_confidence_interval(
        labels,
        predictions,
        metric=lambda truth, pred: float((truth == pred).mean()),
        samples=200,
        seed=17,
    )
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_failure_table_finds_high_confidence_errors():
    labels = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.90, 0.05, 0.03, 0.01, 0.01],
            [0.02, 0.03, 0.90, 0.03, 0.02],
            [0.02, 0.03, 0.80, 0.10, 0.05],
        ]
    )

    table = failure_table(labels, probabilities, paths=["a", "b", "c"])

    assert table["path"].tolist() == ["b"]
    assert table.loc[0, "true_grade"] == 1
    assert table.loc[0, "predicted_grade"] == 2


def test_portable_identifiers_remove_local_directories():
    paths = [
        "/local/cache/images/DR/image_001.png",
        "/tmp/idrid/IDRiD_101.jpg",
    ]

    assert portable_identifiers(paths) == ["image_001.png", "IDRiD_101.jpg"]
