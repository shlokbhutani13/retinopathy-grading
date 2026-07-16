import numpy as np
import torch
from PIL import Image
from torch import nn

from retinopathy.explain import GradCAM, overlay_heatmap


class TinyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 5)

    def forward(self, value):
        value = torch.relu(self.features(value))
        return self.classifier(self.pool(value).flatten(1))


def test_gradcam_and_overlay_match_image_size():
    model = TinyVisionModel()
    image_tensor = torch.rand(1, 3, 32, 32)
    image = Image.new("RGB", (32, 32), (40, 80, 120))

    with GradCAM(model, model.features) as explainer:
        heatmap = explainer(image_tensor, class_index=2)
    overlay = overlay_heatmap(image, heatmap)

    assert heatmap.shape == (32, 32)
    assert 0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1
    assert overlay.size == image.size
    assert np.asarray(overlay).shape == (32, 32, 3)
