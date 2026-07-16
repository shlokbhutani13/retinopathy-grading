from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader

from retinopathy.calibration import (
    expected_calibration_error,
    softmax,
    temperature_scale,
)
from retinopathy.evaluation import evaluate_predictions
from retinopathy.model import build_model
from retinopathy.train import (
    FundusDataset,
    class_weights,
    evaluate_loader,
    train_one_epoch,
)


def run_training(
    *,
    config_path: str | Path,
    split_path: str | Path,
    dataset_root: str | Path,
) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device()
    frame = pd.read_csv(split_path)
    frame["image_path"] = frame["relative_path"].map(lambda value: str(Path(dataset_root) / value))

    loaders = {
        split: DataLoader(
            FundusDataset(
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
    train_labels = frame.loc[frame["split"] == "train", "grade"].to_numpy()
    weights = class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
    )

    best_kappa = float("-inf")
    best_state = None
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        validation = evaluate_loader(model, loaders["validation"], criterion, device)
        probabilities = softmax(validation["logits"])
        metrics = evaluate_predictions(validation["labels"], probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_kappa": metrics["quadratic_weighted_kappa"],
                "validation_macro_f1": metrics["macro_f1"],
            }
        )
        print(json.dumps(history[-1]))
        if metrics["quadratic_weighted_kappa"] > best_kappa:
            best_kappa = float(metrics["quadratic_weighted_kappa"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        scheduler.step()

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    validation = evaluate_loader(model, loaders["validation"], criterion, device)
    temperature, calibrated_validation = temperature_scale(
        validation["logits"],
        validation["labels"],
    )
    test = evaluate_loader(model, loaders["test"], criterion, device)
    calibrated_test = softmax(test["logits"] / temperature)
    metrics = evaluate_predictions(test["labels"], calibrated_test)
    metrics.update(
        {
            "test_loss": float(test["loss"]),
            "temperature": temperature,
            "validation_ece_after_calibration": expected_calibration_error(
                calibrated_validation,
                validation["labels"],
            ),
            "test_ece": expected_calibration_error(calibrated_test, test["labels"]),
            "best_validation_kappa": best_kappa,
            "test_samples": int(len(test["labels"])),
            "device": str(device),
            "history": history,
        }
    )

    model_path = Path(config["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "temperature": temperature,
            "image_size": int(config["image_size"]),
            "grade_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        },
        model_path,
    )
    metrics_path = Path(config["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    save_confusion_matrix(
        test["labels"],
        calibrated_test.argmax(axis=1),
        metrics_path.parent / "confusion_matrix.png",
    )
    return metrics


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=np.arange(5),
        display_labels=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        cmap="Blues",
        normalize="true",
        ax=axis,
        colorbar=False,
    )
    axis.set_title("Normalized held-out test confusion matrix")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
