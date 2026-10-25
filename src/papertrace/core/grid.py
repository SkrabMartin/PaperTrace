"""Paper-grid analysis: rotation of the sheet and the pixel size of one millimetre.

ECG paper is printed at 25 mm/s and 10 mm/mV with a 1 mm grid and heavier lines every 5 mm, so
the grid period in pixels is the scale that converts the image to seconds and millivolts.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

MIN_MM_PERIOD_PX = 3.0
MAX_MM_PERIOD_PX = 14.0
ROTATION_SEARCH_DEG = 15.0
ANALYSIS_WIDTH_PX = 800


def to_gray(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Convert RGB to a float grayscale image."""
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.float64)


def rotate_image(image: NDArray[np.uint8], degrees: float, fill: int = 255) -> NDArray[np.uint8]:
    """Rotate about the image centre by ``degrees`` (counter-clockwise), keeping the size."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(fill, fill, fill),
    )
    return np.asarray(rotated, dtype=np.uint8)


def _sharpness(gray: NDArray[np.float64], degrees: float) -> float:
    """Variance of the row profile after rotating: largest when grid lines are horizontal."""
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (width, height), borderValue=float(gray.mean()))
    margin = int(0.1 * min(height, width))
    centre = rotated[margin : height - margin, margin : width - margin]
    return float(centre.mean(axis=1).var())


def estimate_rotation_deg(
    image: NDArray[np.uint8], search_deg: float = ROTATION_SEARCH_DEG
) -> float:
    """Angle in degrees by which the sheet must be rotated counter-clockwise to be upright.

    The angle that maximises the contrast of the row profile (grid lines horizontal) is searched
    coarsely and then refined, on a downscaled copy of the image.
    """
    gray = to_gray(image)
    scale = min(1.0, ANALYSIS_WIDTH_PX / gray.shape[1])
    small = np.asarray(
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), dtype=np.float64
    )
    coarse = np.arange(-search_deg, search_deg + 1e-9, 0.5)
    best = float(coarse[int(np.argmax([_sharpness(small, a) for a in coarse]))])
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    return float(fine[int(np.argmax([_sharpness(small, a) for a in fine]))])


def _peak_period(profiles: list[NDArray[np.float64]], min_px: float, max_px: float) -> float:
    n_fft = 8 * max(profile.size for profile in profiles)  # zero-padded for finer frequencies
    spectrum = np.zeros(n_fft // 2 + 1)
    for profile in profiles:
        windowed = (profile - profile.mean()) * np.hanning(profile.size)
        spectrum += np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    step = 1.0 / n_fft
    freqs = np.arange(spectrum.size) * step
    valid = np.flatnonzero((freqs >= 1 / max_px) & (freqs <= 1 / min_px))
    peak = int(valid[int(np.argmax(spectrum[valid]))])
    if 0 < peak < spectrum.size - 1:
        left, mid, right = spectrum[peak - 1], spectrum[peak], spectrum[peak + 1]
        denominator = left - 2 * mid + right
        shift = 0.5 * (left - right) / denominator if denominator != 0 else 0.0
        return float(1.0 / ((peak + shift) * step))
    return float(1.0 / freqs[peak])


def estimate_grid_period(
    image: NDArray[np.uint8],
    min_px: float = MIN_MM_PERIOD_PX,
    max_px: float = MAX_MM_PERIOD_PX,
) -> float:
    """Pixels per millimetre from the periodicity of the (upright) paper grid.

    The search band excludes the 5 mm lines' period, which would otherwise be confused with the
    1 mm grid; it covers images from roughly 75 to 350 dpi.
    """
    gray = to_gray(image)
    return _peak_period([gray.mean(axis=0), gray.mean(axis=1)], min_px, max_px)
