from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from retinopathy.quality import crop_retinal_field


def perceptual_hash(path: str | Path) -> np.uint64:
    with Image.open(path) as source:
        image = crop_retinal_field(source.convert("RGB"), image_size=32).convert("L")
    coefficients = dctn(np.asarray(image, dtype=np.float32), norm="ortho")[:8, :8]
    values = coefficients.flatten()[1:]
    bits = values > np.median(values)
    result = 0
    for value in bits:
        result = (result << 1) | int(value)
    return np.uint64(result)


def perceptual_hash_distance(first: np.uint64, second: np.uint64) -> int:
    return int(np.bitwise_count(np.bitwise_xor(first, second)))
