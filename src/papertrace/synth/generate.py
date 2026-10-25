"""Batch orchestration: PTB-XL records in, images + masks + a manifest out."""

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from papertrace.synth.artefacts import apply_artefacts
from papertrace.synth.kit import BatchSpec, KitSetup, run_batch, sample_spec
from papertrace.synth.manifest import SYNTH_COLUMNS, check_folds_inherited
from papertrace.synth.masks import lead_traces, render_lead_id_mask

LARGE_SQUARE_MM = 5.0


def thickness_for(resolution: int) -> int:
    """Mask line width in pixels for an image rendered at ``resolution`` dpi."""
    return max(2, round(resolution / 70))


def lead_boxes(mask: NDArray[np.uint8]) -> dict[str, list[int]]:
    """Bounding box ``[x0, y0, x1, y1]`` of every lead id present in ``mask``."""
    boxes: dict[str, list[int]] = {}
    for lead_id in np.unique(mask):
        if lead_id == 0:
            continue
        ys, xs = np.nonzero(mask == lead_id)
        boxes[str(int(lead_id))] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return boxes


def process_image(
    png: Path,
    config_json: Path,
    image_path: Path,
    mask_path: Path,
    meta_path: Path,
    artefact_seed: int | None,
) -> dict[str, Any]:
    """Turn one generator output into final image, mask and a slim metadata file.

    Args:
        png: Image written by the generator.
        config_json: The generator's configuration with the plotted pixel positions.
        image_path: Where to write the final image.
        mask_path: Where to write the lead id mask.
        meta_path: Where to write the slim metadata JSON.
        artefact_seed: Seed for the extra phone-photo artefacts, or ``None`` for none.

    Returns:
        The metadata that was written.
    """
    config = json.loads(config_json.read_text(encoding="utf-8"))
    image = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
    resolution = int(config["resolution"])
    traces = lead_traces(config)
    mask = render_lead_id_mask(traces, image.shape[:2], thickness_for(resolution))
    applied: list[str] = []
    geometry_modified = False
    if artefact_seed is not None:
        result = apply_artefacts(image, mask, artefact_seed)
        image, mask = result.image, result.mask
        applied, geometry_modified = result.applied, result.geometry_modified
    meta: dict[str, Any] = {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "resolution_dpi": resolution,
        "px_per_mm": None if geometry_modified else float(config["x_grid"]) / LARGE_SQUARE_MM,
        "columns": int(config["number_of_columns_in_image"]),
        "bw": bool(config["bw"]),
        "calibration_pulse": bool(config["dc_pulse"]),
        "printed_text": bool(config["printed_text"]),
        "wrinkles": bool(config["wrinkles"]),
        "generator_rotation_range_deg": int(config["rotate"]),
        "artefacts": applied,
        "geometry_modified": geometry_modified,
        "leads": [
            {
                "name": t.name,
                "id": t.lead_id,
                "start_sample": t.start_sample,
                "end_sample": t.end_sample,
            }
            for t in traces
        ],
        "lead_boxes": lead_boxes(mask),
    }
    for path in (image_path, mask_path, meta_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def select_records(split_manifest: pd.DataFrame, split: str, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` records of one split reproducibly."""
    pool = split_manifest[split_manifest["split"] == split]
    return pool.sample(n=min(n, len(pool)), random_state=seed).sort_values("ecg_id")


def generate_dataset(
    records: pd.DataFrame,
    ptbxl_root: Path,
    out_dir: Path,
    setup: KitSetup,
    variants: int = 1,
    batch_size: int = 10,
    seed: int = 0,
    artefacts: bool = False,
) -> pd.DataFrame:
    """Generate synthetic images, masks and the manifest for ``records``.

    Args:
        records: Rows of the split manifest to render (needs ``filename_hr``).
        ptbxl_root: Directory that holds the PTB-XL records.
        out_dir: Output root; images, masks and metadata go to subfolders, paths in the returned
            manifest are relative to it.
        setup: Generator checkout and interpreter.
        variants: Differently rendered images per record.
        batch_size: Records per generator run (each run shares one parameter set).
        seed: Master seed; everything derives from it.
        artefacts: Also apply the phone-photo artefacts of :mod:`papertrace.synth.artefacts`.

    Returns:
        The synthetic manifest (also checked for fold inheritance).

    Raises:
        RuntimeError: If the generator did not produce an expected output.
    """
    rows: list[dict[str, Any]] = []
    work_root = out_dir / "_work"
    for variant in range(variants):
        order = records.to_dict("records")
        random.Random(seed * 1000 + variant).shuffle(order)
        for batch_index in range(0, len(order), batch_size):
            chunk = order[batch_index : batch_index + batch_size]
            batch_id = f"v{variant}_b{batch_index // batch_size}"
            spec: BatchSpec = sample_spec(
                random.Random(f"{seed}-{batch_id}"),
                seed=seed * 100_000 + variant * 1000 + batch_index,
            )
            batch_in, batch_out = work_root / batch_id / "in", work_root / batch_id / "out"
            batch_in.mkdir(parents=True, exist_ok=True)
            for rec in chunk:
                source = ptbxl_root / rec["filename_hr"]
                for suffix in (".hea", ".dat"):
                    shutil.copy2(source.with_suffix(suffix), batch_in / (source.name + suffix))
            run_batch(setup, spec, batch_in, batch_out)
            for rec in chunk:
                name = Path(rec["filename_hr"]).name
                png, cfg = batch_out / f"{name}-0.png", batch_out / f"{name}-0.json"
                if not (png.exists() and cfg.exists()):
                    raise RuntimeError(f"generator produced no output for ecg_id {rec['ecg_id']}")
                stem_name = f"{int(rec['ecg_id']):05d}_v{variant}"
                art_seed = seed * 7919 + int(rec["ecg_id"]) * 31 + variant if artefacts else None
                paths = {
                    "image_path": f"images/{stem_name}.png",
                    "mask_path": f"masks/{stem_name}_mask.png",
                    "meta_path": f"meta/{stem_name}_meta.json",
                }
                process_image(
                    png,
                    cfg,
                    out_dir / paths["image_path"],
                    out_dir / paths["mask_path"],
                    out_dir / paths["meta_path"],
                    art_seed,
                )
                rows.append(
                    {
                        **paths,
                        "ecg_id": int(rec["ecg_id"]),
                        "patient_id": int(rec["patient_id"]),
                        "strat_fold": int(rec["strat_fold"]),
                        "split": str(rec["split"]),
                        "variant": variant,
                        "seed": spec.seed,
                        "params_json": json.dumps(
                            {**spec.to_dict(), "artefacts": artefacts}, sort_keys=True
                        ),
                    }
                )
            shutil.rmtree(work_root / batch_id)
    shutil.rmtree(work_root, ignore_errors=True)
    manifest = pd.DataFrame(rows, columns=SYNTH_COLUMNS)
    check_folds_inherited(manifest, records[["ecg_id", "patient_id", "strat_fold", "split"]])
    return manifest
