"""Conversion between image coordinates and physical units (mV, seconds, samples).

ECG paper runs at 25 mm/s and 10 mm/mV unless stated otherwise, so once the pixel size of a
millimetre is known the conversions are linear.
"""

import numpy as np
from numpy.typing import NDArray

SPEED_MM_PER_S = 25.0
GAIN_MM_PER_MV = 10.0
MAX_GAP_MM = 3.0


def px_to_mv(
    y_px: NDArray[np.float64],
    baseline_px: float,
    px_per_mm: float,
    gain_mm_per_mv: float = GAIN_MM_PER_MV,
) -> NDArray[np.float64]:
    """Convert row coordinates (y grows downwards) to millivolts relative to ``baseline_px``."""
    return (baseline_px - y_px) / (px_per_mm * gain_mm_per_mv)


def mv_to_px(
    mv: NDArray[np.float64],
    baseline_px: float,
    px_per_mm: float,
    gain_mm_per_mv: float = GAIN_MM_PER_MV,
) -> NDArray[np.float64]:
    """Inverse of :func:`px_to_mv`."""
    return baseline_px - mv * px_per_mm * gain_mm_per_mv


def px_to_seconds(
    x_px: NDArray[np.float64], px_per_mm: float, speed_mm_per_s: float = SPEED_MM_PER_S
) -> NDArray[np.float64]:
    """Convert column distances to seconds."""
    return x_px / (px_per_mm * speed_mm_per_s)


def seconds_to_px(
    seconds: NDArray[np.float64], px_per_mm: float, speed_mm_per_s: float = SPEED_MM_PER_S
) -> NDArray[np.float64]:
    """Inverse of :func:`px_to_seconds`."""
    return seconds * px_per_mm * speed_mm_per_s


def resample_trace(
    x_px: NDArray[np.float64],
    y_mv: NDArray[np.float64],
    px_per_mm: float,
    n_samples: int,
    fs: float = 500.0,
    max_gap_mm: float = MAX_GAP_MM,
) -> NDArray[np.float64]:
    """Resample a per-column trace to ``n_samples`` at ``fs`` Hz, starting at ``x_px[0]``.

    Samples further than ``max_gap_mm`` from any column that has a value, or outside the
    observed range, are NaN rather than interpolated.

    Args:
        x_px: Column positions, increasing.
        y_mv: Trace in mV per column, NaN where nothing was found.
        px_per_mm: Pixel size of one millimetre.
        n_samples: Number of output samples.
        fs: Output sampling frequency in Hz.
        max_gap_mm: Longest gap that may be bridged.
    """
    valid = np.isfinite(y_mv)
    if valid.sum() < 2:
        return np.full(n_samples, np.nan)
    t_valid = px_to_seconds(x_px[valid] - x_px[0], px_per_mm)
    t_out = np.arange(n_samples) / fs
    out = np.interp(t_out, t_valid, y_mv[valid], left=np.nan, right=np.nan)
    position = np.searchsorted(t_valid, t_out)
    last = t_valid.size - 1
    before = t_valid[np.clip(position - 1, 0, last)]
    after = t_valid[np.clip(position, 0, last)]
    nearest = np.minimum(np.abs(t_out - before), np.abs(t_out - after))
    out[nearest > max_gap_mm / SPEED_MM_PER_S] = np.nan
    return np.asarray(out, dtype=np.float64)
