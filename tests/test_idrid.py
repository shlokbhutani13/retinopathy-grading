from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinopathy.idrid import load_idrid_split


def make_idrid_fixture(root: Path) -> None:
    base = root / "B. Disease Grading"
    training = base / "1. Original Images" / "a. Training Set"
    testing = base / "1. Original Images" / "b. Testing Set"
    labels = base / "2. Groundtruths"
    training.mkdir(parents=True)
    testing.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (16, 16), (120, 60, 40)).save(training / "IDRiD_001.jpg")
    Image.new("RGB", (16, 16), (120, 60, 40)).save(testing / "IDRiD_101.jpg")
    pd.DataFrame(
        {"Image name": ["IDRiD_001"], "Retinopathy grade": [3]}
    ).to_csv(labels / "a. IDRiD_Disease Grading_Training Labels.csv", index=False)
    pd.DataFrame(
        {"Image name": ["IDRiD_101"], "Retinopathy grade": [4]}
    ).to_csv(labels / "b. IDRiD_Disease Grading_Testing Labels.csv", index=False)


def test_official_idrid_splits_are_disjoint(tmp_path: Path):
    make_idrid_fixture(tmp_path)

    training = load_idrid_split(tmp_path, "train")
    testing = load_idrid_split(tmp_path, "test")

    assert training.iloc[0].to_dict() == {
        "image_path": str(
            tmp_path
            / "B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg"
        ),
        "image_id": "IDRiD_001",
        "grade": 3,
        "source": "idrid",
        "source_split": "train",
    }
    assert testing.iloc[0]["source_split"] == "test"
    assert set(training["image_id"]).isdisjoint(testing["image_id"])


def test_idrid_loader_rejects_missing_images(tmp_path: Path):
    make_idrid_fixture(tmp_path)
    missing = (
        tmp_path
        / "B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="IDRiD_001"):
        load_idrid_split(tmp_path, "train")


def test_idrid_loader_rejects_unknown_split(tmp_path: Path):
    with pytest.raises(ValueError, match="train or test"):
        load_idrid_split(tmp_path, "validation")
