from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from retinopathy.quality import crop_retinal_field

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_transform(*, image_size: int, training: bool) -> transforms.Compose:
    steps: list[object] = [transforms.Resize((image_size, image_size))]
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)


class FundusDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        image_size: int = 224,
        training: bool = False,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = image_transform(image_size=image_size, training=training)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), int(row["grade"])


class HighResolutionFundusDataset(FundusDataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = crop_retinal_field(
                source.convert("RGB"),
                image_size=self.transform.transforms[0].size[0],
            )
        return self.transform(image), int(row["grade"])


def class_weights(labels: Iterable[int], *, classes: int = 5) -> torch.Tensor:
    counts = np.bincount(np.asarray(list(labels), dtype=int), minlength=classes)
    if np.any(counts == 0):
        raise ValueError("each class must contain at least one training sample")
    weights = counts.sum() / (classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch = labels.shape[0]
        total_loss += float(loss.detach()) * batch
        samples += batch
    return total_loss / max(samples, 1)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    samples = 0
    logits_values = []
    label_values = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        batch = labels.shape[0]
        total_loss += float(loss) * batch
        samples += batch
        logits_values.append(logits.cpu())
        label_values.append(labels.cpu())
    return {
        "loss": total_loss / max(samples, 1),
        "logits": torch.cat(logits_values).numpy(),
        "labels": torch.cat(label_values).numpy(),
    }
