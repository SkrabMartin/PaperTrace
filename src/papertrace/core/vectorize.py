"""Turning an image into a one-dimensional trace.

This is the naive baseline: dark "ink" pixels are separated from the coloured grid, and each
column's trace position is the centroid of the ink in it.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

MAX_INK_LEVEL = 128


def ink_mask(image: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Pixels that are dark in every colour channel, which excludes coloured grid lines.

    The threshold is Otsu's, capped at ``MAX_INK_LEVEL`` so a faint photo does not turn grid
    lines into ink.
    """
    brightest = np.asarray(image.max(axis=2), dtype=np.uint8)
    otsu, _ = cv2.threshold(brightest, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.asarray(brightest <= min(float(otsu), float(MAX_INK_LEVEL)), dtype=np.bool_)


def centroid_trace(ink: NDArray[np.bool_], y_offset: float = 0.0) -> NDArray[np.float64]:
    """Mean row index of the ink in every column; NaN where a column has no ink.

    Args:
        ink: Boolean mask of shape ``(height, width)``.
        y_offset: Added to the result, e.g. the row where ``ink`` starts in the full image.
    """
    counts = ink.sum(axis=0)
    rows = np.arange(ink.shape[0], dtype=np.float64)[:, None]
    totals = (ink * rows).sum(axis=0)
    trace = np.full(ink.shape[1], np.nan)
    present = counts > 0
    trace[present] = totals[present] / counts[present] + y_offset
    return trace
