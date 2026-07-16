from __future__ import annotations

import argparse
import json
from pathlib import Path

from retinopathy.data import (
    create_stratified_splits,
    manifest_from_class_directories,
    remove_conflicting_duplicates,
    summarize_classes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/splits/aptos.csv"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/data_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_root = args.dataset_root / "colored_images"
    manifest = manifest_from_class_directories(image_root)
    duplicate_rows = int(manifest.duplicated("sha256", keep=False).sum())
    unique, conflicting = remove_conflicting_duplicates(manifest)
    splits = create_stratified_splits(unique, seed=args.seed)
    committed = splits[["relative_path", "grade", "split", "sha256"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    committed.to_csv(args.output, index=False)

    summary = summarize_classes(unique)
    report = {
        "dataset": "Diabetic Retinopathy 224x224 (APTOS 2019 derivative)",
        "license": "CC0-1.0",
        "images_before_deduplication": int(len(manifest)),
        "images_after_deduplication": int(len(unique)),
        "duplicate_rows": duplicate_rows,
        "conflicting_hashes_excluded": len(conflicting),
        "class_counts": {str(index): int(row["count"]) for index, row in summary.iterrows()},
        "split_counts": {
            name: int(count) for name, count in splits["split"].value_counts().items()
        },
        "seed": args.seed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
