from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def load_idrid_split(
    root: Path,
    split: Literal["train", "test"],
) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    base = Path(root) / "B. Disease Grading"
    if split == "train":
        image_directory = base / "1. Original Images" / "a. Training Set"
        labels_path = (
            base
            / "2. Groundtruths"
            / "a. IDRiD_Disease Grading_Training Labels.csv"
        )
    else:
        image_directory = base / "1. Original Images" / "b. Testing Set"
        labels_path = (
            base
            / "2. Groundtruths"
            / "b. IDRiD_Disease Grading_Testing Labels.csv"
        )

    labels = pd.read_csv(labels_path)
    result = pd.DataFrame(
        {
            "image_id": labels["Image name"].astype(str),
            "grade": labels["Retinopathy grade"].astype(int),
        }
    )
    result["image_path"] = [
        str(image_directory / f"{image_id}.jpg")
        for image_id in result["image_id"]
    ]
    missing = [path for path in result["image_path"] if not Path(path).is_file()]
    if missing:
        names = ", ".join(Path(path).stem for path in missing[:3])
        raise FileNotFoundError(f"missing IDRiD images: {names}")
    result["source"] = "idrid"
    result["source_split"] = split
    return result[["image_path", "image_id", "grade", "source", "source_split"]]
