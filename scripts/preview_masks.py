"""Write overlays of lead masks on synthetic images for a visual check.

Usage: ``uv run python scripts/preview_masks.py output/synth -n 5 --seed 0``
Each preview shows the image with its mask drawn in colour (one colour per lead id) and is saved
to ``<synth dir>/preview/``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

PALETTE = np.array(
    [
        (0, 0, 0),
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
        (0, 128, 128), (170, 110, 40), (128, 0, 0),
    ],
    dtype=np.uint8,
)  # fmt: skip


def overlay(image: NDArray[np.uint8], mask: NDArray[np.uint8], width: int) -> Image.Image:
    """Return the image with every lead drawn in its own colour, scaled to ``width`` pixels."""
    out = image.copy()
    lines = mask > 0
    out[lines] = PALETTE[np.minimum(mask[lines], len(PALETTE) - 1)]
    picture = Image.fromarray(out)
    height = round(picture.height * width / picture.width)
    return picture.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    """Pick random images from a synthetic manifest and write overlay previews."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synth_dir", type=Path)
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1600)
    args = parser.parse_args()
    manifest = pd.read_parquet(args.synth_dir / "synth_manifest.parquet")
    sample = manifest.sample(n=min(args.n, len(manifest)), random_state=args.seed)
    target = args.synth_dir / "preview"
    target.mkdir(exist_ok=True)
    for row in sample.to_dict("records"):
        image = np.asarray(Image.open(args.synth_dir / row["image_path"]).convert("RGB"))
        mask = np.asarray(Image.open(args.synth_dir / row["mask_path"]))
        path = target / f"{Path(row['image_path']).stem}_overlay.png"
        overlay(image, mask, args.width).save(path)
        print(path)  # noqa: T201


if __name__ == "__main__":
    main()
