from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader

from retinopathy.analysis import (
    bootstrap_confidence_interval,
    clustered_bootstrap_confidence_interval,
    failure_table,
)
from retinopathy.calibration import expected_calibration_error
from retinopathy.evaluation import evaluate_predictions
from retinopathy.finetune import load_ordinal_checkpoint
from retinopathy.ordinal import cumulative_logits_to_probabilities
from retinopathy.ordinal_pipeline import evaluate_ordinal_loader
from retinopathy.pipeline import save_confusion_matrix
from retinopathy.train import HighResolutionFundusDataset


def promotion_decision(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    baseline_aptos = baseline["aptos"]
    candidate_aptos = candidate["aptos"]
    baseline_severe = baseline["idrid"]["per_class_recall"][3]
    candidate_severe = candidate["idrid"]["per_class_recall"][3]
    checks = {
        "idrid_severe_recall": candidate_severe > baseline_severe,
        "aptos_kappa": (
            candidate_aptos["quadratic_weighted_kappa"]
            >= baseline_aptos["quadratic_weighted_kappa"] - 0.02
        ),
        "aptos_referable_auroc": (
            candidate_aptos["referable_auroc"]
            >= baseline_aptos["referable_auroc"] - 0.01
        ),
    }
    promote = all(checks.values())
    return {
        "promote": promote,
        "checks": checks,
        "summary": (
            "candidate meets the predeclared promotion criteria"
            if promote
            else "candidate remains an experiment because a promotion criterion failed"
        ),
    }


def evaluate_ordinal_checkpoint(
    *,
    model_path: Path,
    frame: pd.DataFrame,
    dataset_name: str,
    output_prefix: str,
    artifact_directory: Path,
    device: torch.device,
    batch_size: int = 12,
    group_column: str | None = None,
) -> dict[str, object]:
    model, metadata = load_ordinal_checkpoint(model_path, device)
    model.eval()
    loader = DataLoader(
        HighResolutionFundusDataset(
            frame,
            image_size=int(metadata["image_size"]),
            training=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    result = evaluate_ordinal_loader(
        model,
        loader,
        nn.BCEWithLogitsLoss(),
        device,
    )
    probabilities = cumulative_logits_to_probabilities(
        result["logits"] / float(metadata["temperature"])
    )
    labels = result["labels"]
    predictions = probabilities.argmax(axis=1)
    metrics = evaluate_predictions(labels, probabilities)
    metrics.update(
        {
            "dataset": dataset_name,
            "samples": int(len(labels)),
            "expected_calibration_error": expected_calibration_error(
                probabilities,
                labels,
            ),
            "kappa_95_ci": bootstrap_confidence_interval(
                labels,
                predictions,
                metric=lambda truth, pred: float(
                    cohen_kappa_score(truth, pred, weights="quadratic")
                ),
                samples=1000,
                seed=42,
            ),
        }
    )
    if group_column is not None:
        if group_column not in frame:
            raise ValueError(f"missing group column: {group_column}")
        metrics["kappa_95_ci"] = clustered_bootstrap_confidence_interval(
            labels,
            predictions,
            frame[group_column].to_numpy(),
            metric=lambda truth, pred: float(
                cohen_kappa_score(truth, pred, weights="quadratic")
            ),
            samples=1000,
            seed=42,
        )
        metrics["confidence_interval_unit"] = group_column
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / f"{output_prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    save_confusion_matrix(
        labels,
        predictions,
        artifact_directory / f"{output_prefix}_confusion_matrix.png",
    )
    identifiers = (
        frame["image_id"].astype(str).tolist()
        if "image_id" in frame
        else [Path(path).name for path in frame["image_path"]]
    )
    failure_table(
        labels,
        probabilities,
        paths=identifiers,
    ).to_csv(
        artifact_directory / f"{output_prefix}_high_confidence_errors.csv",
        index=False,
    )
    return metrics
