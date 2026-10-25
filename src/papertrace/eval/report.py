"""Scoring the baseline digitiser on a synthetic dataset and summarising the results."""

import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from papertrace.core.baseline import TOTAL_SAMPLES, digitize_baseline
from papertrace.core.image import load_image
from papertrace.core.layout import LEAD_NAMES, cells_from_entries
from papertrace.data.ptbxl import load_signal
from papertrace.eval.metrics import aligned_estimate, lead_snr_both, pearson, rmse, summarize

_LEAD_INDEX = {name.lower(): index for index, name in enumerate(LEAD_NAMES)}


def masked_reference(
    signal: NDArray[np.float64],
    lead_names: Sequence[str],
    plotted: Sequence[Mapping[str, Any]],
) -> NDArray[np.float64]:
    """The ground truth for an image: the record's signal, NaN where nothing was plotted.

    Args:
        signal: ``(samples, leads)`` in mV, in the order of ``lead_names``.
        lead_names: Lead names of the columns of ``signal``, any letter case.
        plotted: Plotted leads with ``name``, ``start_sample`` and ``end_sample``.

    Returns:
        ``(TOTAL_SAMPLES, 12)`` in the order of ``LEAD_NAMES``.
    """
    column = {name.lower(): index for index, name in enumerate(lead_names)}
    reference = np.full((TOTAL_SAMPLES, len(LEAD_NAMES)), np.nan)
    for entry in plotted:
        name = str(entry["name"]).lower()
        start, end = int(entry["start_sample"]), int(entry["end_sample"])
        reference[start:end, _LEAD_INDEX[name]] = signal[start:end, column[name]]
    return reference


@dataclass(frozen=True)
class ImageScore:
    """Scores of one image, per lead in the order of ``LEAD_NAMES``."""

    ecg_id: int
    layout: str
    resolution: int
    snr: list[float]
    snr_upstream: list[float]
    pearson: list[float]
    rmse_mv: list[float]
    scale_error: float | None
    coverage: float
    ink_fraction: float


def score_image(job: tuple[str, dict[str, Any], str, str]) -> ImageScore:
    """Digitise one synthetic image and score it against its PTB-XL record."""
    synth_dir, row, ptbxl_root, filename_hr = job
    meta = json.loads((Path(synth_dir) / row["meta_path"]).read_text(encoding="utf-8"))
    params = json.loads(row["params_json"])
    image = load_image(Path(synth_dir) / row["image_path"])
    cells = cells_from_entries(meta["leads"], meta["columns"])
    output = digitize_baseline(image, cells, meta["columns"])
    signal, names = load_signal(Path(ptbxl_root), filename_hr)
    reference = masked_reference(signal, names, meta["leads"])
    snr, snr_upstream, corr, error = [], [], [], []
    for j in range(len(LEAD_NAMES)):
        both = lead_snr_both(reference[:, j], output.signal[:, j])
        snr.append(both[0])
        snr_upstream.append(both[1])
        aligned = aligned_estimate(reference[:, j], output.signal[:, j])
        corr.append(pearson(reference[:, j], aligned))
        error.append(rmse(reference[:, j], aligned))
    true_scale = meta["px_per_mm"]
    return ImageScore(
        ecg_id=int(row["ecg_id"]),
        layout=str(params["layout"]),
        resolution=int(params["resolution"]),
        snr=snr,
        snr_upstream=snr_upstream,
        pearson=corr,
        rmse_mv=error,
        scale_error=None if true_scale is None else output.px_per_mm / true_scale - 1,
        coverage=float(np.isfinite(output.signal).mean()),
        ink_fraction=output.ink_fraction,
    )


def _pooled(scores: Sequence[ImageScore], field: str) -> dict[str, float]:
    return summarize([v for s in scores for v in getattr(s, field)])


def _by_tercile(scores: Sequence[ImageScore]) -> dict[str, Any]:
    """SNR split by the share of dark pixels (the more, the more grid was mistaken for ink)."""
    if len(scores) < 3:
        return {}
    cuts = np.percentile([s.ink_fraction for s in scores], [100 / 3, 200 / 3])
    groups = {
        "low": [s for s in scores if s.ink_fraction <= cuts[0]],
        "middle": [s for s in scores if cuts[0] < s.ink_fraction <= cuts[1]],
        "high": [s for s in scores if s.ink_fraction > cuts[1]],
    }
    return {
        name: {
            "ink_fraction_range": [
                min(s.ink_fraction for s in group),
                max(s.ink_fraction for s in group),
            ],
            "snr_db": _pooled(group, "snr"),
        }
        for name, group in groups.items()
        if group
    }


def build_report(scores: Sequence[ImageScore]) -> dict[str, Any]:
    """Summaries per lead, overall and by layout and resolution."""
    report: dict[str, Any] = {
        "n_images": len(scores),
        "overall": {
            "snr_db": _pooled(scores, "snr"),
            "snr_db_upstream_alignment": _pooled(scores, "snr_upstream"),
            "pearson": _pooled(scores, "pearson"),
            "rmse_mv": _pooled(scores, "rmse_mv"),
        },
        "per_lead": {
            name: {
                "snr_db": summarize([s.snr[j] for s in scores]),
                "snr_db_upstream_alignment": summarize([s.snr_upstream[j] for s in scores]),
                "pearson": summarize([s.pearson[j] for s in scores]),
                "rmse_mv": summarize([s.rmse_mv[j] for s in scores]),
            }
            for j, name in enumerate(LEAD_NAMES)
        },
        "scale_relative_error": summarize(
            [s.scale_error for s in scores if s.scale_error is not None]
        ),
        "by_layout": {
            layout: _pooled([s for s in scores if s.layout == layout], "snr")
            for layout in sorted({s.layout for s in scores})
        },
        "by_ink_fraction_tercile": _by_tercile(scores),
        "by_resolution_dpi": {
            str(dpi): _pooled([s for s in scores if s.resolution == dpi], "snr")
            for dpi in sorted({s.resolution for s in scores})
        },
    }
    return report


def evaluate_dataset(
    synth_dir: Path,
    ptbxl_root: Path,
    split_manifest: pd.DataFrame,
    limit: int | None = None,
    workers: int | None = None,
) -> tuple[dict[str, Any], list[ImageScore]]:
    """Run the baseline over a synthetic dataset and summarise the scores.

    Args:
        synth_dir: Output directory of ``papertrace synth``.
        ptbxl_root: PTB-XL directory (for the reference signals).
        split_manifest: The split manifest (supplies each record's file name).
        limit: Evaluate only the first ``limit`` images.
        workers: Parallel processes; default is the CPU count capped at 8.
    """
    synth = pd.read_parquet(synth_dir / "synth_manifest.parquet")
    if limit is not None:
        synth = synth.head(limit)
    filenames = split_manifest.set_index("ecg_id")["filename_hr"]
    jobs = [
        (str(synth_dir), row, str(ptbxl_root), str(filenames[row["ecg_id"]]))
        for row in synth.to_dict("records")
    ]
    count = workers or min(os.cpu_count() or 1, 8)
    with ProcessPoolExecutor(max_workers=count) as pool:
        scores = list(pool.map(score_image, jobs, chunksize=1))
    return build_report(scores), scores
