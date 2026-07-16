from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinopathy.data import (
    GRADE_NAMES,
    create_stratified_splits,
    image_sha256,
    remove_conflicting_duplicates,
    summarize_classes,
    validate_manifest,
)


def write_image(path: Path, color: tuple[int, int, int] = (20, 80, 120)) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def test_manifest_accepts_five_grades_and_existing_images(tmp_path: Path):
    rows = []
    for grade in range(5):
        path = tmp_path / f"eye-{grade}.png"
        write_image(path, (grade * 30, 70, 110))
        rows.append({"image_path": str(path), "grade": grade})

    validated = validate_manifest(pd.DataFrame(rows))

    assert validated["grade"].tolist() == [0, 1, 2, 3, 4]
    assert GRADE_NAMES[4] == "Proliferative DR"


def test_manifest_rejects_unknown_grades_and_missing_images(tmp_path: Path):
    manifest = pd.DataFrame([{"image_path": str(tmp_path / "missing.png"), "grade": 5}])

    with pytest.raises(ValueError, match="grade"):
        validate_manifest(manifest)


def test_stratified_splits_are_disjoint_and_reproducible(tmp_path: Path):
    rows = []
    for grade in range(5):
        for index in range(10):
            path = tmp_path / f"{grade}-{index}.png"
            write_image(path, (grade * 30, index * 10, 100))
            rows.append({"image_path": str(path), "grade": grade})
    manifest = pd.DataFrame(rows)

    first = create_stratified_splits(manifest, seed=19)
    second = create_stratified_splits(manifest, seed=19)

    assert first.equals(second)
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first.groupby(["split", "grade"]).size().min() >= 1


def test_class_summary_and_hashes_are_stable(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    write_image(first)
    write_image(second)
    manifest = pd.DataFrame(
        [{"image_path": str(first), "grade": 0}, {"image_path": str(second), "grade": 1}]
    )

    summary = summarize_classes(manifest)

    assert summary.loc[0, "count"] == 1
    assert image_sha256(first) == image_sha256(second)


def test_conflicting_duplicate_hashes_are_removed_entirely():
    manifest = pd.DataFrame(
        [
            {"image_path": "a.png", "grade": 0, "sha256": "same"},
            {"image_path": "b.png", "grade": 1, "sha256": "same"},
            {"image_path": "c.png", "grade": 2, "sha256": "unique"},
        ]
    )

    cleaned, conflicting_hashes = remove_conflicting_duplicates(manifest)

    assert cleaned["image_path"].tolist() == ["c.png"]
    assert conflicting_hashes == {"same"}
