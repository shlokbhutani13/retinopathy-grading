from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score

from retinopathy.analysis import bootstrap_confidence_interval, failure_table
from retinopathy.calibration import expected_calibration_error
from retinopathy.evaluation import evaluate_predictions
from retinopathy.ordinal import OrdinalClassifier, cumulative_logits_to_probabilities
from retinopathy.pipeline import save_confusion_matrix, select_device
from retinopathy.quality import crop_retinal_field
from retinopathy.train import image_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/retinopathy_ordinal_384.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/idrid_metrics.json"),
    )
    args = parser.parse_args()
    frame = load_idrid_manifest(args.idrid_root)
    artifact = torch.load(args.model, map_location="cpu", weights_only=False)
    device = select_device()
    model = OrdinalClassifier(pretrained=False)
    model.load_state_dict(artifact["model_state"])
    model.to(device).eval()
    transform = image_transform(image_size=int(artifact["image_size"]), training=False)
    logits_values = []
    with torch.no_grad():
        for path in frame["image_path"]:
            with Image.open(path) as source:
                image = crop_retinal_field(
                    source.convert("RGB"),
                    image_size=int(artifact["image_size"]),
                )
            tensor = transform(image).unsqueeze(0).to(device)
            logits_values.append(model(tensor).cpu().numpy()[0])
    logits = np.asarray(logits_values) / float(artifact["temperature"])
    probabilities = cumulative_logits_to_probabilities(logits)
    labels = frame["grade"].to_numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = evaluate_predictions(labels, probabilities)
    metrics.update(
        {
            "dataset": "IDRiD disease grading",
            "license": "CC-BY-4.0",
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    save_confusion_matrix(
        labels,
        predictions,
        args.output.parent / "idrid_confusion_matrix.png",
    )
    failure_table(
        labels,
        probabilities,
        paths=frame["image_path"].tolist(),
    ).to_csv(args.output.parent / "idrid_high_confidence_errors.csv", index=False)
    print(json.dumps(metrics, indent=2))


def load_idrid_manifest(root: Path) -> pd.DataFrame:
    grading = root / "B. Disease Grading"
    parts = []
    for split, label_name, image_name in (
        (
            "training",
            "a. IDRiD_Disease Grading_Training Labels.csv",
            "a. Training Set",
        ),
        (
            "testing",
            "b. IDRiD_Disease Grading_Testing Labels.csv",
            "b. Testing Set",
        ),
    ):
        labels = pd.read_csv(grading / "2. Groundtruths" / label_name)
        labels = labels.loc[:, ["Image name", "Retinopathy grade"]].copy()
        image_directory = grading / "1. Original Images" / image_name
        labels["image_path"] = [
            str(image_directory / f"{value}.jpg") for value in labels["Image name"]
        ]
        labels["grade"] = labels["Retinopathy grade"].astype(int)
        labels["source_split"] = split
        parts.append(labels)
    frame = pd.concat(parts, ignore_index=True)
    missing = [path for path in frame["image_path"] if not Path(path).is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} IDRiD images")
    return frame


if __name__ == "__main__":
    main()
