from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from retinopathy.evaluation import evaluate_predictions
from retinopathy.idrid import load_idrid_split
from retinopathy.ordinal import (
    OrdinalClassifier,
    cumulative_logits_to_probabilities,
    ordinal_temperature_scale,
)
from retinopathy.ordinal_pipeline import (
    evaluate_ordinal_loader,
    resolve_high_resolution_paths,
    train_ordinal_epoch,
)
from retinopathy.pipeline import select_device, set_seed
from retinopathy.train import HighResolutionFundusDataset


def class_aware_sampler(
    labels: np.ndarray,
    *,
    seed: int,
) -> WeightedRandomSampler:
    values = np.asarray(labels, dtype=int)
    counts = np.bincount(values, minlength=5)
    if np.any(counts == 0):
        raise ValueError("every grade must contain at least one training image")
    class_weights = len(values) / (5 * counts)
    weights = torch.tensor(class_weights[values], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(values),
        replacement=True,
        generator=generator,
    )


def load_ordinal_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[OrdinalClassifier, dict[str, object]]:
    metadata = torch.load(path, map_location=device, weights_only=False)
    model = OrdinalClassifier(pretrained=False)
    model.load_state_dict(metadata["model_state"])
    model.to(device)
    return model, metadata


def run_idrid_finetuning(
    *,
    config_path: Path,
    aptos_split_path: Path,
    aptos_root: Path,
    idrid_root: Path,
) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text())
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device()

    aptos = resolve_high_resolution_paths(
        pd.read_csv(aptos_split_path),
        image_directory=aptos_root,
    )
    aptos["source"] = "aptos"
    idrid_training = load_idrid_split(idrid_root, "train")
    training = pd.concat(
        [aptos.loc[aptos["split"] == "train"], idrid_training],
        ignore_index=True,
        sort=False,
    )
    validation = aptos.loc[aptos["split"] == "validation"].reset_index(drop=True)

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    workers = int(config["num_workers"])
    training_dataset = HighResolutionFundusDataset(
        training,
        image_size=image_size,
        training=True,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        sampler=class_aware_sampler(training["grade"].to_numpy(), seed=seed),
        num_workers=workers,
    )
    validation_loader = DataLoader(
        HighResolutionFundusDataset(
            validation,
            image_size=image_size,
            training=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model, source_metadata = load_ordinal_checkpoint(
        Path(config["source_model_path"]),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()

    initial = evaluate_ordinal_loader(
        model,
        validation_loader,
        criterion,
        device,
    )
    initial_metrics = evaluate_predictions(
        initial["labels"],
        cumulative_logits_to_probabilities(
            initial["logits"] / float(source_metadata.get("temperature", 1.0))
        ),
    )
    best_kappa = float(initial_metrics["quadratic_weighted_kappa"])
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "validation_loss": float(initial["loss"]),
            "validation_kappa": best_kappa,
            "validation_macro_f1": float(initial_metrics["macro_f1"]),
        }
    ]

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_ordinal_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device,
        )
        result = evaluate_ordinal_loader(
            model,
            validation_loader,
            criterion,
            device,
        )
        metrics = evaluate_predictions(
            result["labels"],
            cumulative_logits_to_probabilities(result["logits"]),
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": float(result["loss"]),
            "validation_kappa": float(metrics["quadratic_weighted_kappa"]),
            "validation_macro_f1": float(metrics["macro_f1"]),
        }
        history.append(record)
        print(json.dumps(record))
        if record["validation_kappa"] > best_kappa:
            best_kappa = record["validation_kappa"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.to(device)
    calibrated = evaluate_ordinal_loader(
        model,
        validation_loader,
        criterion,
        device,
    )
    temperature, _ = ordinal_temperature_scale(
        calibrated["logits"],
        calibrated["labels"],
    )
    model_path = Path(config["candidate_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "temperature": temperature,
            "image_size": image_size,
            "architecture": "efficientnet_b0_ordinal",
            "fine_tuned_with": "IDRiD official training split",
        },
        model_path,
    )
    report = {
        "device": str(device),
        "training_images": len(training),
        "aptos_training_images": int((training["source"] == "aptos").sum()),
        "idrid_training_images": int((training["source"] == "idrid").sum()),
        "best_epoch": best_epoch,
        "best_validation_kappa": best_kappa,
        "temperature": temperature,
        "history": history,
    }
    metrics_path = Path(config["training_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
