from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retinopathy.deepdrid import (
    cross_dataset_hash_audit,
    load_deepdrid_evaluation,
)


def write_labels_xlsx(path: Path, rows: list[tuple[str, int]]) -> None:
    strings = ["image_id", "DR_Levels", *[image_id for image_id, _ in rows]]
    shared = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + "".join(f"<si><t>{value}</t></si>" for value in strings)
        + "</sst>"
    )
    sheet_rows = [
        '<row r="1"><c r="A1" t="s"><v>0</v></c>'
        '<c r="B1" t="s"><v>1</v></c></row>'
    ]
    for index, (_, grade) in enumerate(rows, start=2):
        string_index = index
        sheet_rows.append(
            f'<row r="{index}"><c r="A{index}" t="s"><v>{string_index}</v></c>'
            f'<c r="B{index}"><v>{grade}</v></c></row>'
        )
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData></worksheet>"
    )
    path.parent.mkdir(parents=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def make_deepdrid_fixture(root: Path, rows: list[tuple[str, int]]) -> None:
    evaluation = (
        root
        / "regular_fundus_images"
        / "Online-Challenge1&2-Evaluation"
    )
    write_labels_xlsx(evaluation / "Challenge1_labels.xlsx", rows)
    for index, (image_id, _) in enumerate(rows):
        patient_id = image_id.split("_", maxsplit=1)[0]
        directory = evaluation / "Images" / patient_id
        directory.mkdir(parents=True, exist_ok=True)
        values = np.full((24, 24, 3), (110 + index, 60, 35), dtype=np.uint8)
        values[4:20, 4:20] += index
        Image.fromarray(values).save(directory / f"{image_id}.jpg")


def test_loads_official_evaluation_with_patient_ids(tmp_path: Path):
    rows = [("347_l1", 1), ("347_l2", 1), ("347_r1", 4), ("347_r2", 4)]
    make_deepdrid_fixture(tmp_path, rows)

    result = load_deepdrid_evaluation(tmp_path)

    assert result.columns.tolist() == [
        "image_path",
        "image_id",
        "patient_id",
        "grade",
        "source",
        "source_split",
    ]
    assert result["image_id"].tolist() == [row[0] for row in rows]
    assert result["patient_id"].tolist() == ["347"] * 4
    assert result["grade"].tolist() == [1, 1, 4, 4]
    assert set(result["source_split"]) == {"online_evaluation"}


def test_rejects_duplicate_image_ids(tmp_path: Path):
    make_deepdrid_fixture(tmp_path, [("347_l1", 1), ("347_l1", 2)])

    with pytest.raises(ValueError, match="duplicate image IDs"):
        load_deepdrid_evaluation(tmp_path)


def test_rejects_missing_images(tmp_path: Path):
    make_deepdrid_fixture(tmp_path, [("347_l1", 1)])
    next(tmp_path.rglob("347_l1.jpg")).unlink()

    with pytest.raises(FileNotFoundError, match="347_l1"):
        load_deepdrid_evaluation(tmp_path)


def test_hash_audit_stops_on_exact_cross_dataset_duplicate(tmp_path: Path):
    image = Image.new("RGB", (32, 32), (120, 65, 40))
    target_path = tmp_path / "target.jpg"
    reference_path = tmp_path / "reference.jpg"
    image.save(target_path)
    reference_path.write_bytes(target_path.read_bytes())
    target = pd.DataFrame({"image_id": ["target"], "image_path": [target_path]})
    reference = pd.DataFrame(
        {"image_id": ["reference"], "image_path": [reference_path]}
    )

    with pytest.raises(ValueError, match="exact duplicate"):
        cross_dataset_hash_audit(target, {"reference": reference})


def test_hash_audit_reports_close_perceptual_matches(tmp_path: Path):
    values = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:64, :64]
    values[(xx - 32) ** 2 + (yy - 32) ** 2 <= 25**2] = (140, 75, 40)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(values).save(first)
    values[30, 30] = (141, 75, 40)
    Image.fromarray(values).save(second)
    target = pd.DataFrame({"image_id": ["first"], "image_path": [first]})
    reference = pd.DataFrame({"image_id": ["second"], "image_path": [second]})

    result = cross_dataset_hash_audit(target, {"reference": reference})

    assert result["exact_matches"] == []
    assert result["perceptual_matches"][0]["target_image_id"] == "first"
    assert result["perceptual_matches"][0]["reference_image_id"] == "second"
