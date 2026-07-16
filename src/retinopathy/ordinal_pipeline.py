from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader

from retinopathy.analysis import bootstrap_confidence_interval, failure_table
from retinopathy.calibration import expected_calibration_error
from retinopathy.evaluation import evaluate_predictions
from retinopathy.ordinal import (
    OrdinalClassifier,
    cumulative_logits_to_probabilities,
    ordinal_targets,
    ordinal_temperature_scale,
)
from retinopathy.pipeline import save_confusion_matrix, select_device, set_seed
from retinopathy.train import HighResolutionFundusDataset


def run_ordinal_training(
    *,
    config_path: str | Path,
    split_path: str | Path,
    image_directory: str | Path,
) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    set_seed(int(config["seed"]))
    device = select_device()
    frame = resolve_high_resolution_paths(
        pd.read_csv(split_path),
        image_directory=Path(image_directory),
    )
    loaders = {
        split: DataLoader(
            HighResolutionFundusDataset(
                frame.loc[frame["split"] == split],
                image_size=int(config["image_size"]),
                training=split == "train",
            ),
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            num_workers=int(config["num_workers"]),
        )
        for split in ("train", "validation", "test")
    }
    model = OrdinalClassifier(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
    )
    positive_weights = ordinal_positive_weights(
        frame.loc[frame["split"] == "train", "grade"].to_numpy()
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)

    best_kappa = float("-inf")
    best_state = None
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_ordinal_epoch(model, loaders["train"], optimizer, criterion, device)
        validation = evaluate_ordinal_loader(model, loaders["validation"], criterion, device)
        probabilities = cumulative_logits_to_probabilities(validation["logits"])
        metrics = evaluate_predictions(validation["labels"], probabilities)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_kappa": metrics["quadratic_weighted_kappa"],
            "validation_macro_f1": metrics["macro_f1"],
        }
        history.append(record)
        print(json.dumps(record))
        if metrics["quadratic_weighted_kappa"] > best_kappa:
            best_kappa = float(metrics["quadratic_weighted_kappa"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        scheduler.step()

    if best_state is None:
        raise RuntimeError("ordinal training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation = evaluate_ordinal_loader(model, loaders["validation"], criterion, device)
    temperature, validation_probabilities = ordinal_temperature_scale(
        validation["logits"],
        validation["labels"],
    )
    test = evaluate_ordinal_loader(model, loaders["test"], criterion, device)
    test_probabilities = cumulative_logits_to_probabilities(test["logits"] / temperature)
    metrics = evaluate_predictions(test["labels"], test_probabilities)
    predictions = test_probabilities.argmax(axis=1)
    metrics.update(
        {
            "test_loss": float(test["loss"]),
            "temperature": temperature,
            "validation_ece_after_calibration": expected_calibration_error(
                validation_probabilities,
                validation["labels"],
            ),
            "test_ece": expected_calibration_error(test_probabilities, test["labels"]),
            "best_validation_kappa": best_kappa,
            "test_samples": int(len(test["labels"])),
            "device": str(device),
            "history": history,
            "kappa_95_ci": bootstrap_confidence_interval(
                test["labels"],
                predictions,
                metric=lambda truth, pred: float(
                    cohen_kappa_score(truth, pred, weights="quadratic")
                ),
                samples=1000,
                seed=int(config["seed"]),
            ),
        }
    )

    model_path = Path(config["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "temperature": temperature,
            "image_size": int(config["image_size"]),
            "architecture": "efficientnet_b0_ordinal",
        },
        model_path,
    )
    artifact_dir = Path(config["metrics_path"]).parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    Path(config["metrics_path"]).write_text(json.dumps(metrics, indent=2) + "\n")
    save_confusion_matrix(
        test["labels"],
        predictions,
        artifact_dir / "ordinal_confusion_matrix.png",
    )
    failure_table(
        test["labels"],
        test_probabilities,
        paths=frame.loc[frame["split"] == "test", "image_path"].tolist(),
    ).to_csv(artifact_dir / "ordinal_high_confidence_errors.csv", index=False)
    return metrics


def resolve_high_resolution_paths(
    frame: pd.DataFrame,
    *,
    image_directory: Path,
) -> pd.DataFrame:
    resolved = frame.copy()
    path_column = (
        "high_relative_path"
        if "high_relative_path" in resolved.columns
        else "relative_path"
    )
    resolved["image_path"] = resolved[path_column].map(
        lambda value: str(image_directory / value)
    )
    missing = [
        path for path in resolved["image_path"] if not Path(path).is_file()
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} high-resolution images")
    return resolved


def ordinal_positive_weights(labels: np.ndarray) -> torch.Tensor:
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    targets = ordinal_targets(labels_tensor)
    positives = targets.sum(dim=0)
    negatives = len(targets) - positives
    return negatives / positives.clamp_min(1)


def train_ordinal_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, ordinal_targets(labels))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        samples += len(labels)
    return total_loss / max(samples, 1)


@torch.no_grad()
def evaluate_ordinal_loader(model, loader, criterion, device) -> dict[str, object]:
    model.eval()
    losses = 0.0
    samples = 0
    logits_values = []
    labels_values = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, ordinal_targets(labels))
        losses += float(loss) * len(labels)
        samples += len(labels)
        logits_values.append(logits.cpu())
        labels_values.append(labels.cpu())
    return {
        "loss": losses / max(samples, 1),
        "logits": torch.cat(logits_values).numpy(),
        "labels": torch.cat(labels_values).numpy(),
    }
