"""Ground-truth masks for synthetic ECG images, rendered from ECG-Image-Kit's geometry.

The generator records, for every plotted lead, the pixel position of each signal sample
(``plotted_pixels`` as ``[y, x]`` with y counted from the top). The mask is a ``uint8`` image of
the same size: 0 is background, 1-12 the standard leads in the order of ``STANDARD_LEADS`` and
13 the rhythm strip. A binary curve mask is ``mask > 0``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
RHYTHM_STRIP_ID = 13
DEFAULT_THICKNESS_PX = 3

_ID_BY_NAME = {name.lower(): number for number, name in enumerate(STANDARD_LEADS, start=1)}


@dataclass(frozen=True)
class LeadTrace:
    """One plotted curve: which lead it is and where its samples lie in the image."""

    name: str
    lead_id: int
    start_sample: int
    end_sample: int
    xy: NDArray[np.float64]  # (n_samples, 2) as (x, y) pixel coordinates, y from the top


def lead_id(name: str) -> int:
    """Return the mask id (1-12) of a standard lead name, case-insensitively."""
    try:
        return _ID_BY_NAME[name.lower()]
    except KeyError:
        raise ValueError(f"unknown lead name: {name!r}") from None


def lead_traces(config: Mapping[str, Any]) -> list[LeadTrace]:
    """Extract all plotted curves from an ECG-Image-Kit config (``--store_config 2``).

    The generator lists the rhythm strip as one more entry that repeats a lead name; an entry
    whose name was already seen is therefore labelled ``RHYTHM_STRIP_ID``.
    """
    traces: list[LeadTrace] = []
    seen: set[str] = set()
    for entry in config["leads"]:
        name = str(entry["lead_name"])
        identifier = RHYTHM_STRIP_ID if name.lower() in seen else lead_id(name)
        seen.add(name.lower())
        yx = np.asarray(entry["plotted_pixels"], dtype=np.float64).reshape(-1, 2)
        traces.append(
            LeadTrace(
                name=name,
                lead_id=identifier,
                start_sample=int(entry["start_sample"]),
                end_sample=int(entry["end_sample"]),
                xy=yx[:, ::-1].copy(),
            )
        )
    return traces


def render_lead_id_mask(
    traces: Sequence[LeadTrace],
    shape: tuple[int, int],
    thickness: int = DEFAULT_THICKNESS_PX,
) -> NDArray[np.uint8]:
    """Draw every trace as a polyline of its lead id.

    Args:
        traces: Curves from :func:`lead_traces`.
        shape: ``(height, width)`` of the image.
        thickness: Line width in pixels.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for trace in traces:
        points = np.rint(trace.xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [points], isClosed=False, color=trace.lead_id, thickness=thickness)
    return mask
