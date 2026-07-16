from __future__ import annotations

import timm
from torch import nn


def build_model(*, pretrained: bool = True, num_classes: int = 5) -> nn.Module:
    return timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )
