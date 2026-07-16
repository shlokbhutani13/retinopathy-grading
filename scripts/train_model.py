from __future__ import annotations

import argparse
import json
from pathlib import Path

from retinopathy.pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/aptos.csv"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    metrics = run_training(
        config_path=args.config,
        split_path=args.splits,
        dataset_root=args.dataset_root,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
