from __future__ import annotations

import kagglehub


def main() -> None:
    path = kagglehub.dataset_download("sovitrath/diabetic-retinopathy-224x224-2019-data")
    print(path)


if __name__ == "__main__":
    main()
