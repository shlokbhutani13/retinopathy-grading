from __future__ import annotations

import numpy as np
import timm
import torch
from scipy.optimize import minimize_scalar
from torch import nn


class OrdinalClassifier(nn.Module):
    def __init__(self, *, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=4,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.backbone(value)


def ordinal_targets(labels: torch.Tensor, *, thresholds: int = 4) -> torch.Tensor:
    boundaries = torch.arange(thresholds, device=labels.device)
    return (labels.unsqueeze(1) > boundaries.unsqueeze(0)).float()


def cumulative_logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("ordinal logits must have shape (samples, 4)")
    cumulative = 1.0 / (1.0 + np.exp(-logits))
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    probabilities = np.column_stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2] - cumulative[:, 3],
            cumulative[:, 3],
        ]
    )
    return np.clip(probabilities, 0.0, 1.0)


def ordinal_temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    def negative_log_likelihood(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = cumulative_logits_to_probabilities(logits / temperature)
        selected = probabilities[np.arange(len(labels)), labels]
        return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(-3.0, 3.0),
        method="bounded",
    )
    temperature = float(np.exp(result.x))
    return temperature, cumulative_logits_to_probabilities(logits / temperature)
