from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retinopathy.matching import perceptual_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-resolution-root", type=Path, required=True)
    parser.add_argument("--high-resolution-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("data/splits/aptos.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/aptos_highres.csv"),
    )
    parser.add_argument("--max-distance", type=int, default=10)
    args = parser.parse_args()

    splits = pd.read_csv(args.splits)
    low_paths = [args.low_resolution_root / value for value in splits["relative_path"]]
    high_paths = sorted(
        path
        for path in args.high_resolution_root.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    low_hashes = np.asarray([perceptual_hash(path) for path in low_paths], dtype=np.uint64)
    high_hashes = np.asarray([perceptual_hash(path) for path in high_paths], dtype=np.uint64)

    low_best, low_distances = nearest_matches(low_hashes, high_hashes)
    high_best, _ = nearest_matches(high_hashes, low_hashes)
    mutual = np.asarray(
        [high_best[high_index] == low_index for low_index, high_index in enumerate(low_best)]
    )
    high_binary = np.asarray(
        [0 if path.parent.name == "No DR" else 1 for path in high_paths]
    )
    grade_binary = (splits["grade"].to_numpy() > 0).astype(int)
    binary_agreement = high_binary[low_best] == grade_binary
    accepted = mutual & binary_agreement & (low_distances <= args.max_distance)

    result = splits.loc[accepted, ["grade", "split", "sha256"]].copy()
    result["image_id"] = splits.loc[accepted, "relative_path"].map(
        lambda value: Path(value).stem
    )
    result["high_relative_path"] = [
        str(high_paths[index].relative_to(args.high_resolution_root))
        for index in low_best[accepted]
    ]
    result["perceptual_hash_distance"] = low_distances[accepted]
    result = result[
        [
            "image_id",
            "high_relative_path",
            "grade",
            "split",
            "sha256",
            "perceptual_hash_distance",
        ]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    report = {
        "low_resolution_images": len(low_paths),
        "high_resolution_images": len(high_paths),
        "accepted_mutual_matches": int(accepted.sum()),
        "binary_label_agreement": float(binary_agreement[accepted].mean()),
        "maximum_distance": args.max_distance,
        "split_counts": result["split"].value_counts().to_dict(),
        "class_counts": {
            str(key): int(value) for key, value in result["grade"].value_counts().items()
        },
    }
    print(json.dumps(report, indent=2))


def nearest_matches(
    source_hashes: np.ndarray,
    target_hashes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    distances = []
    for value in source_hashes:
        values = np.bitwise_count(np.bitwise_xor(target_hashes, value))
        index = int(values.argmin())
        indices.append(index)
        distances.append(int(values[index]))
    return np.asarray(indices), np.asarray(distances)


if __name__ == "__main__":
    main()
