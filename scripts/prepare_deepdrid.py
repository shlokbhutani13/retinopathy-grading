from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from retinopathy.deepdrid import (
    cross_dataset_hash_audit,
    load_deepdrid_evaluation,
)
from retinopathy.idrid import load_idrid_split
from retinopathy.ordinal_pipeline import resolve_high_resolution_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the official DeepDRiD online evaluation split."
    )
    parser.add_argument("--deepdrid-root", type=Path, required=True)
    parser.add_argument("--aptos-root", type=Path, required=True)
    parser.add_argument("--idrid-root", type=Path, required=True)
    parser.add_argument(
        "--aptos-manifest",
        type=Path,
        default=Path("data/splits/aptos_highres.csv"),
    )
    return parser.parse_args()


def repository_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    args = parse_args()
    deepdrid = load_deepdrid_evaluation(args.deepdrid_root)
    if len(deepdrid) != 400 or deepdrid["patient_id"].nunique() != 100:
        raise ValueError(
            "expected the official DeepDRiD evaluation split: 400 images, 100 patients"
        )

    aptos = pd.read_csv(args.aptos_manifest)
    aptos = resolve_high_resolution_paths(
        aptos,
        image_directory=args.aptos_root,
    )
    if "image_id" not in aptos:
        aptos["image_id"] = aptos["image_path"].map(lambda path: Path(path).stem)
    idrid = pd.concat(
        [
            load_idrid_split(args.idrid_root, "train"),
            load_idrid_split(args.idrid_root, "test"),
        ],
        ignore_index=True,
    )
    audit = cross_dataset_hash_audit(
        deepdrid,
        {"aptos": aptos, "idrid": idrid},
    )
    report = {
        "dataset": "DeepDRiD",
        "source_split": "online_evaluation",
        "license": "CC BY-SA 4.0",
        "repository_revision": repository_revision(args.deepdrid_root),
        "images": len(deepdrid),
        "patients": int(deepdrid["patient_id"].nunique()),
        "grade_counts": {
            str(key): int(value)
            for key, value in deepdrid["grade"].value_counts().sort_index().items()
        },
        "cross_dataset_audit": audit,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
