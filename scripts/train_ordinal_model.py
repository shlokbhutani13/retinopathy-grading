from __future__ import annotations

import argparse
import json
from pathlib import Path

from retinopathy.ordinal_pipeline import run_ordinal_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ordinal_384.yaml"))
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/aptos_highres.csv"),
    )
    args = parser.parse_args()
    metrics = run_ordinal_training(
        config_path=args.config,
        split_path=args.splits,
        image_directory=args.image_directory,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
