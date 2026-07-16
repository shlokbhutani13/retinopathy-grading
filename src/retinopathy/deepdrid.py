from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from retinopathy.data import GRADE_NAMES, image_sha256
from retinopathy.matching import perceptual_hash
from retinopathy.quality import crop_retinal_field

_SPREADSHEET_NAMESPACE = {
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
}


def _cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    value = cell.find("x:v", _SPREADSHEET_NAMESPACE)
    if value is None or value.text is None:
        inline = cell.find("x:is/x:t", _SPREADSHEET_NAMESPACE)
        return "" if inline is None or inline.text is None else inline.text
    if cell.get("t") == "s":
        return shared_strings[int(value.text)]
    return value.text


def _read_label_rows(path: Path) -> list[tuple[str, int]]:
    with ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", _SPREADSHEET_NAMESPACE):
                shared_strings.append(
                    "".join(
                        node.text or ""
                        for node in item.findall(".//x:t", _SPREADSHEET_NAMESPACE)
                    )
                )
        sheet = ElementTree.fromstring(
            workbook.read("xl/worksheets/sheet1.xml")
        )

    rows: list[tuple[str, int]] = []
    for row in sheet.findall(".//x:sheetData/x:row", _SPREADSHEET_NAMESPACE):
        values = {
            cell.get("r", "")[:1]: _cell_value(cell, shared_strings)
            for cell in row.findall("x:c", _SPREADSHEET_NAMESPACE)
        }
        if values.get("A") == "image_id":
            continue
        if not values.get("A"):
            continue
        rows.append((values["A"], int(float(values["B"]))))
    return rows


def load_deepdrid_evaluation(root: str | Path) -> pd.DataFrame:
    evaluation = (
        Path(root)
        / "regular_fundus_images"
        / "Online-Challenge1&2-Evaluation"
    )
    labels_path = evaluation / "Challenge1_labels.xlsx"
    rows = _read_label_rows(labels_path)
    image_ids = [image_id for image_id, _ in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("DeepDRiD labels contain duplicate image IDs")

    unknown = sorted({grade for _, grade in rows} - set(GRADE_NAMES))
    if unknown:
        raise ValueError(f"DeepDRiD grade values must be 0 through 4; received {unknown}")

    records = []
    missing = []
    for image_id, grade in rows:
        patient_id = image_id.split("_", maxsplit=1)[0]
        image_path = evaluation / "Images" / patient_id / f"{image_id}.jpg"
        if not image_path.is_file():
            missing.append(image_id)
        records.append(
            {
                "image_path": str(image_path),
                "image_id": image_id,
                "patient_id": patient_id,
                "grade": grade,
                "source": "deepdrid",
                "source_split": "online_evaluation",
            }
        )
    if missing:
        names = ", ".join(missing[:3])
        raise FileNotFoundError(f"missing DeepDRiD images: {names}")
    return pd.DataFrame.from_records(
        records,
        columns=[
            "image_path",
            "image_id",
            "patient_id",
            "grade",
            "source",
            "source_split",
        ],
    )


def _mutual_perceptual_matches(
    target: pd.DataFrame,
    reference: pd.DataFrame,
    reference_source: str,
    *,
    maximum_distance: int = 10,
) -> list[dict[str, object]]:
    target_hashes = np.asarray(
        [perceptual_hash(path) for path in target["image_path"]],
        dtype=np.uint64,
    )
    reference_hashes = np.asarray(
        [perceptual_hash(path) for path in reference["image_path"]],
        dtype=np.uint64,
    )
    if not len(target_hashes) or not len(reference_hashes):
        return []

    distances = np.bitwise_count(
        np.bitwise_xor(target_hashes[:, None], reference_hashes[None, :])
    )
    target_nearest = distances.argmin(axis=1)
    reference_nearest = distances.argmin(axis=0)
    matches = []
    for target_index, reference_index in enumerate(target_nearest):
        distance = int(distances[target_index, reference_index])
        if reference_nearest[reference_index] != target_index:
            continue
        target_path = target.iloc[target_index]["image_path"]
        reference_path = reference.iloc[reference_index]["image_path"]
        pixel_difference = _mean_absolute_image_difference(
            target_path, reference_path
        )
        if distance > maximum_distance and pixel_difference > 1.0:
            continue
        matches.append(
            {
                "reference_source": reference_source,
                "target_image_id": str(target.iloc[target_index]["image_id"]),
                "reference_image_id": str(
                    reference.iloc[reference_index]["image_id"]
                ),
                "distance": distance,
                "mean_absolute_pixel_difference": round(pixel_difference, 4),
            }
        )
    return matches


def _mean_absolute_image_difference(
    first: str | Path,
    second: str | Path,
) -> float:
    arrays = []
    for path in (first, second):
        with Image.open(path) as source:
            image = crop_retinal_field(source.convert("RGB"), image_size=64)
            arrays.append(np.asarray(image, dtype=np.float32))
    return float(np.abs(arrays[0] - arrays[1]).mean())


def cross_dataset_hash_audit(
    target: pd.DataFrame,
    references: dict[str, pd.DataFrame],
) -> dict[str, object]:
    target_hashes = {
        image_sha256(path): str(image_id)
        for image_id, path in zip(
            target["image_id"], target["image_path"], strict=True
        )
    }
    exact_matches = []
    perceptual_matches = []
    reference_counts = {}
    for source, reference in references.items():
        reference_counts[source] = len(reference)
        for image_id, path in zip(
            reference["image_id"], reference["image_path"], strict=True
        ):
            digest = image_sha256(path)
            if digest in target_hashes:
                exact_matches.append(
                    {
                        "reference_source": source,
                        "target_image_id": target_hashes[digest],
                        "reference_image_id": str(image_id),
                    }
                )
        perceptual_matches.extend(
            _mutual_perceptual_matches(target, reference, source)
        )

    if exact_matches:
        first = exact_matches[0]
        raise ValueError(
            "cross-dataset exact duplicate: "
            f"{first['target_image_id']} and {first['reference_image_id']}"
        )
    return {
        "target_images": len(target),
        "reference_images": reference_counts,
        "exact_matches": exact_matches,
        "perceptual_matches": perceptual_matches,
    }
