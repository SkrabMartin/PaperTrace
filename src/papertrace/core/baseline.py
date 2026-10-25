"""Classical baseline digitiser: grid scale, centroid per column, no learned components.

The lead topology (which lead is in which cell and which samples it covers) is an input; all
pixel positions, the scale and the rotation are estimated from the image. This isolates the
quality of calibration and vectorisation from the question of how the layout is found.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from papertrace.core.calibrate import (
    GAIN_MM_PER_MV,
    SPEED_MM_PER_S,
    px_to_mv,
    resample_trace,
)
from papertrace.core.grid import estimate_grid_period, estimate_rotation_deg, rotate_image
from papertrace.core.image import resolution_warnings
from papertrace.core.layout import (
    LEAD_NAMES,
    PAPER_SECONDS,
    LeadCell,
    find_row_bands,
)
from papertrace.core.schemas import DigitizationResult, LeadResult
from papertrace.core.vectorize import centroid_trace, ink_mask

FS = 500
EDGE_MARGIN_MM = 3.0
TOTAL_SAMPLES = int(PAPER_SECONDS * FS)
_LEAD_INDEX = {name.lower(): index for index, name in enumerate(LEAD_NAMES)}


@dataclass(frozen=True)
class BaselineOutput:
    """Signal, summary and the intermediate estimates, for inspection and evaluation."""

    signal: NDArray[np.float64]  # (TOTAL_SAMPLES, 12) in mV, NaN where a lead was not plotted
    result: DigitizationResult
    rotation_deg: float
    px_per_mm: float
    ink_fraction: float  # share of dark pixels; a diagnostic for grid contamination


def digitize_baseline(
    image: NDArray[np.uint8], cells: Sequence[LeadCell], columns: int
) -> BaselineOutput:
    """Digitise a 12-lead ECG image with the classical baseline.

    Args:
        image: RGB image.
        cells: Lead topology, see :func:`papertrace.core.layout.cells_from_entries`.
        columns: Number of lead columns in the layout (4 for 3x4, 2 for 6x2).

    Returns:
        The 12-lead signal sampled at 500 Hz and a summary. A lead that appears both in the grid
        and as a rhythm strip is taken from the rhythm strip, which covers the whole record.
    """
    rotation = estimate_rotation_deg(image)
    upright = rotate_image(image, rotation) if abs(rotation) > 0.05 else image
    px_per_mm = estimate_grid_period(upright)
    ink = ink_mask(upright)
    ink_fraction = float(ink.mean())  # high values mean the grid was picked up as ink
    margin = round(EDGE_MARGIN_MM * px_per_mm)  # page borders and scan edges are not trace
    ink[:margin, :] = False
    ink[ink.shape[0] - margin :, :] = False
    ink[:, :margin] = False
    ink[:, ink.shape[1] - margin :] = False
    n_rows = 1 + max(cell.row for cell in cells)
    bands = find_row_bands(ink, n_rows)
    rhythm_band = bands[-1]
    ink_columns = np.flatnonzero(ink[rhythm_band[0] : rhythm_band[1]].any(axis=0))
    paper_px = PAPER_SECONDS * SPEED_MM_PER_S * px_per_mm
    x_right = float(ink_columns[-1]) if ink_columns.size else float(ink.shape[1] - 1)
    x_left = x_right - paper_px
    signal = np.full((TOTAL_SAMPLES, len(LEAD_NAMES)), np.nan)
    leads: list[LeadResult] = []
    for cell in sorted(cells, key=lambda c: c.rhythm):  # rhythm strip last: it wins for lead II
        samples = cell.end_sample - cell.start_sample
        start_px = x_left + cell.start_sample / FS * SPEED_MM_PER_S * px_per_mm
        end_px = start_px + samples / FS * SPEED_MM_PER_S * px_per_mm
        x0, x1 = max(0, round(start_px)), min(ink.shape[1], round(end_px))
        top, bottom = bands[cell.row]
        trace = centroid_trace(ink[top:bottom, x0:x1], y_offset=top)
        coverage = float(np.isfinite(trace).mean()) if trace.size else 0.0
        flags = [] if coverage > 0.9 else ["gaps"]
        if np.isfinite(trace).any():
            baseline = float(np.nanmedian(trace))
            y_mv = px_to_mv(trace, baseline, px_per_mm)
            resampled = resample_trace(
                np.arange(x0, x1, dtype=np.float64), y_mv, px_per_mm, samples
            )
            signal[cell.start_sample : cell.end_sample, _LEAD_INDEX[cell.name.lower()]] = resampled
        leads.append(
            LeadResult(
                name=cell.name,
                start_s=cell.start_sample / FS,
                duration_s=samples / FS,
                confidence=coverage,
                flags=[*flags, "rhythm_strip"] if cell.rhythm else flags,
            )
        )
    result = DigitizationResult(
        fs=FS,
        px_per_mm=px_per_mm,
        scale_source="grid",
        leads=leads,
        global_confidence=float(np.mean([lead.confidence for lead in leads])),
        warnings=resolution_warnings(image),
    )
    return BaselineOutput(
        signal=signal,
        result=result,
        rotation_deg=rotation,
        px_per_mm=px_per_mm,
        ink_fraction=ink_fraction,
    )


__all__ = ["GAIN_MM_PER_MV", "BaselineOutput", "digitize_baseline"]
