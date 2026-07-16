from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image

from retinopathy.deepdrid import load_deepdrid_evaluation
from retinopathy.pipeline import select_device
from retinopathy.promotion import evaluate_ordinal_checkpoint
from retinopathy.quality import assess_image_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen ordinal model on DeepDRiD."
    )
    parser.add_argument("--deepdrid-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/retinopathy_ordinal_384.pt"),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/deepdrid"),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def image_quality_summary(paths: list[str]) -> dict[str, object]:
    accepted = 0
    rejection_reasons: Counter[str] = Counter()
    for path in paths:
        with Image.open(path) as image:
            assessment = assess_image_quality(image)
        accepted += int(assessment["acceptable"])
        rejection_reasons.update(assessment["reasons"])
    return {
        "accepted": accepted,
        "total": len(paths),
        "accepted_fraction": round(accepted / max(len(paths), 1), 4),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def main() -> None:
    args = parse_args()
    frame = load_deepdrid_evaluation(args.deepdrid_root)
    if len(frame) != 400 or frame["patient_id"].nunique() != 100:
        raise ValueError(
            "expected the official DeepDRiD evaluation split: 400 images, 100 patients"
        )

    model_hash_before = file_sha256(args.model)
    metrics = evaluate_ordinal_checkpoint(
        model_path=args.model,
        frame=frame,
        dataset_name="DeepDRiD online evaluation",
        output_prefix="deepdrid",
        artifact_directory=args.artifacts,
        device=select_device(),
        batch_size=args.batch_size,
        group_column="patient_id",
    )
    model_hash_after = file_sha256(args.model)
    if model_hash_after != model_hash_before:
        raise RuntimeError("the frozen model checkpoint changed during evaluation")

    metrics.update(
        {
            "license": "CC BY-SA 4.0",
            "source_split": "online_evaluation",
            "patients": int(frame["patient_id"].nunique()),
            "model_sha256": model_hash_before,
            "source_revision": repository_revision(args.deepdrid_root),
            "image_quality": image_quality_summary(frame["image_path"].tolist()),
        }
    )
    output = args.artifacts / "deepdrid_metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
