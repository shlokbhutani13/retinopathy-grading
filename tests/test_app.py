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


def test_analysis_rejects_unusable_image():
    image = Image.new("RGB", (32, 32), "black")

    with pytest.raises(ValueError, match="quality"):
        analyze_image(image, FakePredictor())


def test_analysis_returns_explanation_and_research_warning():
    values = np.zeros((128, 128, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    values[(xx - 64) ** 2 + (yy - 64) ** 2 <= 54**2] = (140, 75, 45)
    values[60:68, 60:68] = 230
    image = Image.fromarray(values)

    summary, probabilities, overlay = analyze_image(image, FakePredictor())

    assert "Moderate" in summary
    assert "professional examination" in summary
    assert "not a diagnosis" in summary
    assert probabilities["Moderate"] == 0.72
    assert overlay.size == image.size


def test_demo_builds_without_model_artifact():
    demo = build_demo(predictor=None)

    assert demo is not None
