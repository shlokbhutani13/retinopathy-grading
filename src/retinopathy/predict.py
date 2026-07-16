from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from retinopathy.calibration import softmax
from retinopathy.data import GRADE_NAMES
from retinopathy.explain import GradCAM, overlay_heatmap
from retinopathy.model import build_model
from retinopathy.ordinal import OrdinalClassifier, cumulative_logits_to_probabilities
from retinopathy.quality import crop_retinal_field
from retinopathy.train import image_transform


def format_prediction(
    probabilities: np.ndarray,
    *,
    confidence_threshold: float = 0.55,
) -> dict[str, object]:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (5,):
        raise ValueError("probabilities must contain five values")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-5):
        raise ValueError("probabilities must be non-negative and sum to one")

    grade = int(np.argmax(values))
    confidence = float(values[grade])
    return {
        "grade": grade,
        "grade_name": GRADE_NAMES[grade],
        "referable_dr": grade >= 2,
        "confidence": round(confidence, 4),
        "low_confidence": confidence < confidence_threshold,
        "probabilities": {
            GRADE_NAMES[index]: round(float(value), 4) for index, value in enumerate(values)
        },
    }


class RetinopathyPredictor:
    def __init__(
        self,
        artifact_path: str,
        *,
        device: str | None = None,
    ):
        self.device = torch.device(device or _default_device())
        artifact = torch.load(artifact_path, map_location=self.device, weights_only=False)
        self.image_size = int(artifact.get("image_size", 224))
        self.temperature = float(artifact.get("temperature", 1.0))
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(artifact["model_state"])
        self.model.to(self.device).eval()
        self.transform = image_transform(image_size=self.image_size, training=False)

    def predict(self, image: Image.Image) -> tuple[dict[str, object], Image.Image]:
        source = image.convert("RGB")
        tensor = self.transform(source).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).cpu().numpy() / self.temperature
        probabilities = softmax(logits)[0]
        result = format_prediction(probabilities)
        target_layer = self.model.conv_head
        with GradCAM(self.model, target_layer) as explainer:
            heatmap = explainer(tensor, class_index=int(result["grade"]))
        return result, overlay_heatmap(source, heatmap)


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class OrdinalRetinopathyPredictor:
    def __init__(self, artifact_path: str, *, device: str | None = None):
        self.device = torch.device(device or _default_device())
        artifact = torch.load(artifact_path, map_location=self.device, weights_only=False)
        self.image_size = int(artifact.get("image_size", 384))
        self.temperature = float(artifact.get("temperature", 1.0))
        self.model = OrdinalClassifier(pretrained=False)
        self.model.load_state_dict(artifact["model_state"])
        self.model.to(self.device).eval()
        self.transform = image_transform(image_size=self.image_size, training=False)

    def predict(self, image: Image.Image) -> tuple[dict[str, object], Image.Image]:
        source = crop_retinal_field(image.convert("RGB"), image_size=self.image_size)
        tensor = self.transform(source).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).cpu().numpy() / self.temperature
        probabilities = cumulative_logits_to_probabilities(logits)[0]
        result = format_prediction(probabilities)
        with GradCAM(self.model, self.model.backbone.conv_head) as explainer:
            heatmap = explainer(tensor, class_index=min(int(result["grade"]), 3))
        return result, overlay_heatmap(source, heatmap)
