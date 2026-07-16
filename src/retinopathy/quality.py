from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def crop_retinal_field(image: Image.Image, *, image_size: int = 384) -> Image.Image:
    source = image.convert("RGB")
    values = np.asarray(source)
    luminance = values.mean(axis=2)
    foreground = luminance > 12
    if foreground.any():
        rows, columns = np.where(foreground)
        top, bottom = int(rows.min()), int(rows.max()) + 1
        left, right = int(columns.min()), int(columns.max()) + 1
        width, height = right - left, bottom - top
        side = max(width, height)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        left = max(0, int(round(center_x - side / 2)))
        top = max(0, int(round(center_y - side / 2)))
        right = min(source.width, left + side)
        bottom = min(source.height, top + side)
        source = source.crop((left, top, right, bottom))
    return source.resize((image_size, image_size), Image.Resampling.LANCZOS)


def assess_image_quality(image: Image.Image) -> dict[str, object]:
    values = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
    luminance = values.mean(axis=2)
    coverage = float((luminance > 12).mean())
    foreground = luminance[luminance > 12]
    red, green, blue = values.transpose(2, 0, 1)
    retinal_color = (
        (red > green * 1.12)
        & (green > blue * 1.05)
        & (luminance > 12)
    )
    retinal_color_fraction = float(
        retinal_color.sum() / max((luminance > 12).sum(), 1)
    )
    brightness = float(foreground.mean()) if foreground.size else 0.0
    contrast = float(foreground.std()) if foreground.size else 0.0
    edges = np.asarray(
        Image.fromarray(luminance.astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
        dtype=np.float32,
    )
    sharpness = float(edges.var())

    reasons = []
    if coverage < 0.30:
        reasons.append("retinal field")
    if retinal_color_fraction < 0.04:
        reasons.append("retinal color")
    if brightness < 25:
        reasons.append("brightness")
    if brightness > 230:
        reasons.append("brightness")
    if contrast < 9:
        reasons.append("contrast")
    if sharpness < 45:
        reasons.append("sharpness")

    return {
        "acceptable": not reasons,
        "reasons": reasons,
        "retinal_coverage": round(coverage, 4),
        "retinal_color_fraction": round(retinal_color_fraction, 4),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
    }
