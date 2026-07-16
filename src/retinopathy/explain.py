from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._capture_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _capture_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, *, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        image_tensor = image_tensor.detach().requires_grad_(True)
        logits = self.model(image_tensor)
        logits[:, class_index].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model features")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam -= cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        return cam.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.38,
) -> Image.Image:
    source = image.convert("RGB")
    normalized = np.clip(np.asarray(heatmap, dtype=float), 0.0, 1.0)
    red = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    red[..., 0] = (normalized * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(red).resize(source.size, Image.Resampling.BILINEAR)
    return Image.blend(source, heatmap_image, alpha)
