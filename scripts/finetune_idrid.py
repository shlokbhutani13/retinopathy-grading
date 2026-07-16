from __future__ import annotations

import argparse
import json
from pathlib import Path

from retinopathy.finetune import run_idrid_finetuning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid-root", type=Path, required=True)
    parser.add_argument("--aptos-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/idrid_finetune.yaml"),
    )
    parser.add_argument(
        "--aptos-splits",
        type=Path,
        default=Path("data/splits/aptos_highres.csv"),
    )
    args = parser.parse_args()
    report = run_idrid_finetuning(
        config_path=args.config,
        aptos_split_path=args.aptos_splits,
        aptos_root=args.aptos_root,
        idrid_root=args.idrid_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
