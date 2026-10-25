"""Lead layout of a printed ECG: which lead sits in which cell and which seconds it covers.

Lead names are the standard twelve. ``cells_from_entries`` reads the topology of a layout (not
any pixel position) from a list of plotted leads, such as the one ECG-Image-Kit stores; pixel
positions are estimated from the image by :func:`find_row_bands`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PAPER_SECONDS = 10.0


@dataclass(frozen=True)
class LeadCell:
    """One plotted lead.

    ``row`` counts from the top of the sheet; the rhythm strip, if any, is the last row.
    """

    name: str
    row: int
    col: int
    start_sample: int
    end_sample: int
    rhythm: bool


def cells_from_entries(entries: Sequence[Mapping[str, Any]], columns: int) -> list[LeadCell]:
    """Build the cell grid from plotted leads listed bottom row first, left to right.

    A repeated lead name marks the rhythm strip, which is the last entry. Each entry needs
    ``name``, ``start_sample`` and ``end_sample``.

    Raises:
        ValueError: If the non-rhythm leads do not fill whole rows.
    """
    seen: set[str] = set()
    main: list[Mapping[str, Any]] = []
    rhythm: Mapping[str, Any] | None = None
    for entry in entries:
        key = str(entry["name"]).lower()
        if key in seen:
            rhythm = entry
        else:
            seen.add(key)
            main.append(entry)
    if len(main) % columns:
        raise ValueError(f"{len(main)} leads do not fill rows of {columns} columns")
    rows = len(main) // columns
    cells = [
        LeadCell(
            name=str(entry["name"]),
            row=rows - 1 - index // columns,
            col=index % columns,
            start_sample=int(entry["start_sample"]),
            end_sample=int(entry["end_sample"]),
            rhythm=False,
        )
        for index, entry in enumerate(main)
    ]
    if rhythm is not None:
        cells.append(
            LeadCell(
                name=str(rhythm["name"]),
                row=rows,
                col=0,
                start_sample=int(rhythm["start_sample"]),
                end_sample=int(rhythm["end_sample"]),
                rhythm=True,
            )
        )
    return cells


def find_row_bands(ink: NDArray[np.bool_], n_rows: int) -> list[tuple[int, int]]:
    """Split the image height into ``n_rows`` horizontal bands, one per row of leads.

    Rows are located as peaks of the smoothed ink density along the vertical axis; each band
    extends halfway to its neighbours. If fewer peaks than rows are found, rows are spaced evenly
    between the first and the last row that contains ink.
    """
    height = ink.shape[0]
    profile = gaussian_filter1d(ink.sum(axis=1).astype(np.float64), sigma=max(2.0, height * 0.004))
    peaks, props = find_peaks(profile, distance=max(1, height // (2 * n_rows)), height=0)
    if len(peaks) >= n_rows:
        strongest = np.argsort(props["peak_heights"])[::-1][:n_rows]
        centres = np.sort(peaks[strongest]).astype(np.float64)
    else:
        occupied = np.flatnonzero(profile > 0.05 * profile.max())
        centres = np.linspace(occupied[0], occupied[-1], n_rows)
    middles = (centres[:-1] + centres[1:]) / 2
    if n_rows > 1:
        top = max(0.0, centres[0] - (middles[0] - centres[0]))
        bottom = min(float(height), centres[-1] + (centres[-1] - middles[-1]))
    else:
        top, bottom = 0.0, float(height)
    edges = np.concatenate(([top], middles, [bottom]))
    return [(round(a), round(b)) for a, b in pairwise(edges)]
