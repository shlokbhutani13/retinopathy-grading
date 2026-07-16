from __future__ import annotations

import argparse
import json
from pathlib import Path

from retinopathy.idrid import load_idrid_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid-root", type=Path, required=True)
    args = parser.parse_args()

    report = {}
    for split in ("train", "test"):
        frame = load_idrid_split(args.idrid_root, split)
        report[split] = {
            "images": len(frame),
            "class_counts": frame["grade"].value_counts().sort_index().to_dict(),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
