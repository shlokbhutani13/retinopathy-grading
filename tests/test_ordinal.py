import numpy as np
import torch

from retinopathy.ordinal import (
    OrdinalClassifier,
    cumulative_logits_to_probabilities,
    ordinal_targets,
)


def test_ordinal_targets_encode_four_severity_thresholds():
    targets = ordinal_targets(torch.tensor([0, 2, 4]))

    assert targets.tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]


def test_cumulative_logits_create_valid_five_grade_probabilities():
    logits = np.array([[4.0, 2.0, -1.0, -3.0]])

    probabilities = cumulative_logits_to_probabilities(logits)

    assert probabilities.shape == (1, 5)
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities.argmax(axis=1).tolist() == [2]


def test_ordinal_model_returns_four_logits():
    model = OrdinalClassifier(pretrained=False)
    logits = model(torch.zeros(2, 3, 96, 96))

    assert logits.shape == (2, 4)
