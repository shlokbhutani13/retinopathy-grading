from pathlib import Path

import numpy as np
from PIL import Image

from retinopathy.matching import perceptual_hash, perceptual_hash_distance


def test_perceptual_hash_matches_resized_copy(tmp_path: Path):
    values = np.zeros((128, 128, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    mask = (xx - 64) ** 2 + (yy - 64) ** 2 < 48**2
    values[mask, 0] = (90 + xx.repeat(128, axis=0)[mask] // 2).astype(np.uint8)
    values[mask, 1] = (45 + yy.repeat(128, axis=1)[mask] // 3).astype(np.uint8)
    values[mask, 2] = 35
    values[58:70, 30:42] = 220
    values[62:65, 40:105] = (70, 25, 20)
    source = Image.fromarray(values)
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    source.save(first)
    source.resize((512, 512)).save(second)

    distance = perceptual_hash_distance(
        perceptual_hash(first),
        perceptual_hash(second),
    )

    assert distance <= 10
