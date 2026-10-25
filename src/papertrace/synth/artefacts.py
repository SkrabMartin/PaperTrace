"""Phone-photo artefacts that ECG-Image-Kit does not provide, applied to image and mask together.

Photometric effects (blur, compression, shadow, flash glare, exposure) change only the image.
Geometric ones (perspective, edge crop) also move the mask, which is resampled with nearest
neighbour so lead ids stay intact. After a geometric change the pixel scale of the paper grid is
no longer uniform, which callers must record.
"""

import random
from dataclasses import dataclass

import albumentations as alb
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArtefactResult:
    """Augmented image and mask plus what was done."""

    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    geometry_modified: bool
    applied: list[str]


def _photometric(seed: int) -> alb.Compose:
    return alb.Compose(
        [
            alb.MotionBlur(blur_limit=(3, 7), p=0.3),
            alb.RandomShadow(shadow_intensity_range=(0.1, 0.3), p=0.4),
            alb.RandomSunFlare(src_radius=150, p=0.1),
            alb.RandomBrightnessContrast(p=0.5),
            alb.ImageCompression(quality_range=(30, 85), p=0.8),
        ],
        seed=seed,
    )


def apply_artefacts(image: NDArray[np.uint8], mask: NDArray[np.uint8], seed: int) -> ArtefactResult:
    """Apply a random set of phone-photo artefacts.

    Args:
        image: RGB ``uint8`` image of shape ``(h, w, 3)``.
        mask: Lead id mask of shape ``(h, w)``.
        seed: Seed; the same inputs and seed always give the same result.

    Returns:
        The transformed pair; ``geometry_modified`` is true if the mask was moved.
    """
    rng = random.Random(seed)
    applied: list[str] = []
    geometry: alb.BasicTransform | None = None
    if rng.random() < 0.5:
        geometry = alb.Perspective(scale=(0.02, 0.05), p=1.0)
        applied.append("perspective")
    elif rng.random() < 0.5:
        geometry = alb.CropAndPad(percent=(-0.06, -0.01), keep_size=True, p=1.0)
        applied.append("edge_crop")
    if geometry is not None:
        out = alb.Compose([geometry], seed=seed)(image=image, mask=mask)
        image, mask = out["image"], out["mask"]
    out = _photometric(seed)(image=image, mask=mask)
    applied.append("photometric")
    return ArtefactResult(
        image=np.asarray(out["image"], dtype=np.uint8),
        mask=np.asarray(out["mask"], dtype=np.uint8),
        geometry_modified=geometry is not None,
        applied=applied,
    )
