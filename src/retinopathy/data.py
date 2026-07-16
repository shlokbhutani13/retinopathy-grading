from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"image_path", "grade"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    validated = manifest.loc[:, ["image_path", "grade"]].copy()
    validated["grade"] = pd.to_numeric(validated["grade"], errors="raise").astype(int)
    unknown = sorted(set(validated["grade"]) - set(GRADE_NAMES))
    if unknown:
        raise ValueError(f"grade values must be 0 through 4; received {unknown}")

    for value in validated["image_path"]:
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"image does not exist: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"image cannot be read: {path}") from exc

    return validated.reset_index(drop=True)


def create_stratified_splits(
    manifest: pd.DataFrame,
    *,
    seed: int = 42,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1")

    train, holdout = train_test_split(
        manifest,
        test_size=validation_size + test_size,
        stratify=manifest["grade"],
        random_state=seed,
    )
    relative_test_size = test_size / (validation_size + test_size)
    validation, test = train_test_split(
        holdout,
        test_size=relative_test_size,
        stratify=holdout["grade"],
        random_state=seed,
    )

    parts = []
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        part = frame.copy()
        part["split"] = name
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["split", "grade", "image_path"])
        .reset_index(drop=True)
    )


def summarize_classes(manifest: pd.DataFrame) -> pd.DataFrame:
    summary = (
        manifest.groupby("grade", observed=False)
        .size()
        .reindex(GRADE_NAMES, fill_value=0)
        .rename("count")
        .to_frame()
    )
    summary["grade_name"] = [GRADE_NAMES[grade] for grade in summary.index]
    summary["fraction"] = summary["count"] / max(int(summary["count"].sum()), 1)
    return summary


def image_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_from_class_directories(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    folder_grades = {
        "No_DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferate_DR": 4,
    }
    rows = []
    for folder, grade in folder_grades.items():
        directory = root / folder
        if not directory.is_dir():
            raise ValueError(f"missing class directory: {directory}")
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            rows.append(
                {
                    "image_path": str(path),
                    "relative_path": str(path.relative_to(root.parent)),
                    "grade": grade,
                    "sha256": image_sha256(path),
                }
            )
    if not rows:
        raise ValueError(f"no supported images found under {root}")
    return pd.DataFrame(rows)


def remove_conflicting_duplicates(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    label_counts = manifest.groupby("sha256")["grade"].nunique()
    conflicting_hashes = set(label_counts.loc[label_counts > 1].index)
    cleaned = manifest.loc[~manifest["sha256"].isin(conflicting_hashes)].copy()
    cleaned = cleaned.drop_duplicates("sha256", keep="first").reset_index(drop=True)
    return cleaned, conflicting_hashes
