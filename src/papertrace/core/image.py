"""Loading images safely and normalising their orientation."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

MAX_PIXELS = 50_000_000
LOW_RESOLUTION_PX = 800


def load_image(path: Path) -> NDArray[np.uint8]:
    """Read an image as RGB ``uint8``, applying its EXIF orientation.

    Raises:
        ValueError: If the image has more than ``MAX_PIXELS`` pixels (decompression-bomb guard).
    """
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError(f"image too large: {image.width}x{image.height} pixels")
        oriented = ImageOps.exif_transpose(image) or image
        return np.asarray(oriented.convert("RGB"), dtype=np.uint8)


def resolution_warnings(image: NDArray[np.uint8]) -> list[str]:
    """Warnings about an image that is probably too small to digitise reliably."""
    if image.shape[1] < LOW_RESOLUTION_PX:
        return [f"low resolution: {image.shape[1]} px wide (< {LOW_RESOLUTION_PX})"]
    return []
