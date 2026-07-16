from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from retinopathy.idrid import load_idrid_split
from retinopathy.ordinal_pipeline import resolve_high_resolution_paths
from retinopathy.pipeline import select_device
from retinopathy.promotion import evaluate_ordinal_checkpoint, promotion_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid-root", type=Path, required=True)
    parser.add_argument("--aptos-root", type=Path, required=True)
    parser.add_argument(
        "--aptos-splits",
        type=Path,
        default=Path("data/splits/aptos_highres.csv"),
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=Path("models/retinopathy_ordinal_384.pt"),
    )
    parser.add_argument(
        "--candidate-model",
        type=Path,
        default=Path("models/retinopathy_ordinal_idrid.pt"),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("artifacts/idrid_finetune"),
    )
    args = parser.parse_args()

    aptos = resolve_high_resolution_paths(
        pd.read_csv(args.aptos_splits),
        image_directory=args.aptos_root,
    )
    aptos_test = aptos.loc[aptos["split"] == "test"].reset_index(drop=True)
    idrid_test = load_idrid_split(args.idrid_root, "test")
    device = select_device()

    comparison = {}
    for name, path in (
        ("baseline", args.baseline_model),
        ("candidate", args.candidate_model),
    ):
        comparison[name] = {
            "aptos": evaluate_ordinal_checkpoint(
                model_path=path,
                frame=aptos_test,
                dataset_name="APTOS held-out test",
                output_prefix=f"{name}_aptos",
                artifact_directory=args.artifact_directory,
                device=device,
            ),
            "idrid": evaluate_ordinal_checkpoint(
                model_path=path,
                frame=idrid_test,
                dataset_name="IDRiD official test",
                output_prefix=f"{name}_idrid",
                artifact_directory=args.artifact_directory,
                device=device,
            ),
        }
    comparison["decision"] = promotion_decision(
        comparison["baseline"],
        comparison["candidate"],
    )
    output = args.artifact_directory / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
