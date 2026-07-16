import numpy as np
import pytest
from PIL import Image

from app import analyze_image, build_demo


class FakePredictor:
    def predict(self, image):
        return {
            "grade": 2,
            "grade_name": "Moderate",
            "referable_dr": True,
            "confidence": 0.72,
            "low_confidence": False,
            "probabilities": {
                "No DR": 0.05,
                "Mild": 0.08,
                "Moderate": 0.72,
                "Severe": 0.10,
                "Proliferative DR": 0.05,
            },
        }, image


def test_analysis_rejects_missing_image():
    with pytest.raises(ValueError, match="retinal"):
        analyze_image(None, FakePredictor())


def test_analysis_returns_explanation_and_research_warning():
    image = Image.fromarray(np.full((32, 32, 3), 90, dtype=np.uint8))

    summary, probabilities, overlay = analyze_image(image, FakePredictor())

    assert "Moderate" in summary
    assert "professional examination" in summary
    assert "not a diagnosis" in summary
    assert probabilities["Moderate"] == 0.72
    assert overlay.size == image.size


def test_demo_builds_without_model_artifact():
    demo = build_demo(predictor=None)

    assert demo is not None
