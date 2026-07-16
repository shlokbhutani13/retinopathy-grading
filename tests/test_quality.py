import numpy as np
from PIL import Image, ImageFilter

from retinopathy.quality import assess_image_quality, crop_retinal_field


def retinal_image(size: int = 128) -> Image.Image:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= (size * 0.42) ** 2
    canvas[mask] = (140, 75, 45)
    canvas[size // 2 - 4 : size // 2 + 4, size // 2 - 4 : size // 2 + 4] = 230
    return Image.fromarray(canvas)


def test_crop_retinal_field_removes_black_border_and_resizes():
    cropped = crop_retinal_field(retinal_image(), image_size=96)

    assert cropped.size == (96, 96)
    assert np.asarray(cropped).mean() > np.asarray(retinal_image()).mean()


def test_quality_accepts_retinal_like_image():
    result = assess_image_quality(retinal_image())

    assert result["acceptable"] is True
    assert result["retinal_coverage"] > 0.4


def test_quality_rejects_blank_and_blurry_images():
    blank = Image.new("RGB", (128, 128), "black")
    blurry = retinal_image().filter(ImageFilter.GaussianBlur(12))

    blank_result = assess_image_quality(blank)
    blurry_result = assess_image_quality(blurry)

    assert blank_result["acceptable"] is False
    assert "retinal field" in blank_result["reasons"]
    assert blurry_result["acceptable"] is False
    assert "sharpness" in blurry_result["reasons"]


def test_quality_rejects_non_retinal_screenshot():
    values = np.full((128, 128, 3), 245, dtype=np.uint8)
    values[:28] = (35, 29, 39)
    values[45:50, 15:110] = (150, 150, 150)
    values[65:70, 15:95] = (40, 90, 190)

    result = assess_image_quality(Image.fromarray(values))

    assert result["acceptable"] is False
    assert "retinal color" in result["reasons"]
