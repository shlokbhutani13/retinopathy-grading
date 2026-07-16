import numpy as np
import torch

from retinopathy.model import build_model
from retinopathy.predict import format_prediction


def test_model_returns_five_logits():
    model = build_model(pretrained=False)
    output = model(torch.zeros(2, 3, 224, 224))

    assert output.shape == (2, 5)


def test_prediction_formats_grade_and_referable_result():
    result = format_prediction(np.array([0.02, 0.03, 0.70, 0.20, 0.05]))

    assert result["grade"] == 2
    assert result["grade_name"] == "Moderate"
    assert result["referable_dr"] is True
    assert result["confidence"] == 0.70
    assert result["low_confidence"] is False


def test_prediction_flags_uncertainty():
    result = format_prediction(np.array([0.24, 0.22, 0.20, 0.18, 0.16]))

    assert result["low_confidence"] is True
