from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn

from retinopathy.train import (
    FundusDataset,
    class_weights,
    evaluate_loader,
    train_one_epoch,
)


def create_manifest(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for grade in range(5):
        for index in range(2):
            path = tmp_path / f"{grade}-{index}.png"
            Image.new("RGB", (32, 32), (grade * 35, index * 50, 100)).save(path)
            rows.append({"image_path": str(path), "grade": grade})
    return pd.DataFrame(rows)


def test_dataset_returns_normalized_tensor_and_label(tmp_path: Path):
    dataset = FundusDataset(create_manifest(tmp_path), image_size=32, training=False)

    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label == 0


def test_class_weights_favor_rare_classes():
    weights = class_weights([0, 0, 0, 1, 2, 3, 4], classes=5)

    assert weights[0] < weights[4]
    assert weights.shape == (5,)


def test_training_smoke_updates_and_evaluates(tmp_path: Path):
    dataset = FundusDataset(create_manifest(tmp_path), image_size=32, training=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 5))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    loss = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    evaluation = evaluate_loader(model, loader, criterion, torch.device("cpu"))

    assert loss > 0
    assert evaluation["loss"] > 0
    assert evaluation["logits"].shape == (10, 5)
    assert evaluation["labels"].shape == (10,)
