import json
import random
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from papertrace.data.manifest import build_manifest
from papertrace.data.splits import LeakageError
from papertrace.synth.artefacts import apply_artefacts
from papertrace.synth.generate import (
    generate_dataset,
    lead_boxes,
    process_image,
    select_records,
    thickness_for,
)
from papertrace.synth.kit import GENERATOR_SUBDIR, BatchSpec, KitSetup, sample_spec
from papertrace.synth.manifest import (
    SYNTH_COLUMNS,
    check_folds_inherited,
    read_synth_manifest,
    write_synth_manifest,
)
from papertrace.synth.masks import (
    RHYTHM_STRIP_ID,
    lead_id,
    lead_traces,
    render_lead_id_mask,
)


def _config(width: int = 120, height: int = 80) -> dict[str, Any]:
    """A minimal ECG-Image-Kit config: leads I and II plus a rhythm strip repeating II."""
    return {
        "width": width,
        "height": height,
        "resolution": 200,
        "x_grid": 39.37,
        "number_of_columns_in_image": 4,
        "bw": False,
        "dc_pulse": True,
        "printed_text": False,
        "wrinkles": False,
        "rotate": 0,
        "leads": [
            {
                "lead_name": "I",
                "start_sample": 0,
                "end_sample": 3,
                "plotted_pixels": [[10, 5], [10, 20], [30, 40]],  # [y, x]
            },
            {
                "lead_name": "AVR",
                "start_sample": 0,
                "end_sample": 2,
                "plotted_pixels": [[50, 10], [50, 60]],
            },
            {
                "lead_name": "II",
                "start_sample": 0,
                "end_sample": 2,
                "plotted_pixels": [[60, 10], [60, 100]],
            },
            {
                "lead_name": "II",
                "start_sample": 0,
                "end_sample": 2,
                "plotted_pixels": [[70, 10], [70, 100]],
            },
        ],
    }


def test_lead_id_is_case_insensitive_and_strict() -> None:
    assert lead_id("I") == 1
    assert lead_id("AVR") == lead_id("aVR") == 4
    assert lead_id("v6") == 12
    with pytest.raises(ValueError, match="unknown lead name"):
        lead_id("X")


def test_traces_swap_to_xy_and_mark_rhythm_strip() -> None:
    traces = lead_traces(_config())
    assert [t.lead_id for t in traces] == [1, 4, 2, RHYTHM_STRIP_ID]
    np.testing.assert_array_equal(traces[0].xy[0], [5, 10])  # [y, x] -> (x, y)


def test_mask_draws_lead_ids_on_the_curve() -> None:
    mask = render_lead_id_mask(lead_traces(_config()), (80, 120), thickness=1)
    assert mask.dtype == np.uint8
    assert mask[10, 5] == 1
    assert mask[10, 20] == 1
    assert mask[50, 30] == 4
    assert mask[70, 50] == RHYTHM_STRIP_ID
    assert mask[0, 0] == 0
    assert set(np.unique(mask)) == {0, 1, 2, 4, RHYTHM_STRIP_ID}


def test_mask_ignores_points_outside_the_image() -> None:
    config = _config()
    config["leads"][0]["plotted_pixels"] = [[-50, -50], [500, 500]]
    mask = render_lead_id_mask(lead_traces(config), (80, 120))
    assert mask.shape == (80, 120)


def test_lead_boxes_cover_each_id() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:5, 3:9] = 7
    assert lead_boxes(mask) == {"7": [3, 2, 8, 4]}


@pytest.mark.parametrize(("dpi", "expected"), [(100, 2), (200, 3), (300, 4)])
def test_mask_thickness_grows_with_resolution(dpi: int, expected: int) -> None:
    assert thickness_for(dpi) == expected


def test_batch_args_store_geometry_and_never_print_the_path() -> None:
    spec = BatchSpec("3x4", 200, wrinkles=False, rotate=0, noise=0, seed=1)
    args = spec.to_args()
    assert {"--store_config", "--lead_bbox", "--lead_name_bbox"} <= set(args)
    assert not any("print_header" in arg for arg in args)
    assert "--augment" not in args
    assert "--wrinkles" not in args
    augmented = BatchSpec("6x2", 100, wrinkles=True, rotate=5, noise=10, seed=1).to_args()
    assert {"--augment", "--wrinkles", "--deterministic_noise"} <= set(augmented)
    assert augmented[augmented.index("--num_columns") + 1] == "2"


def test_sample_spec_is_deterministic() -> None:
    assert sample_spec(random.Random(3), 3) == sample_spec(random.Random(3), 3)


def test_artefacts_are_deterministic_and_keep_lead_ids() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[40:44, 10:150] = 3
    first = apply_artefacts(image, mask, seed=5)
    second = apply_artefacts(image, mask, seed=5)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.image.shape == image.shape
    assert first.mask.shape == mask.shape
    assert set(np.unique(first.mask)) <= {0, 3}


def test_photometric_only_leaves_the_mask_unchanged() -> None:
    image = np.full((60, 80, 3), 200, dtype=np.uint8)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:23, 5:70] = 2
    results = [apply_artefacts(image, mask, seed=s) for s in range(30)]
    plain = [r for r in results if not r.geometry_modified]
    moved = [r for r in results if r.geometry_modified]
    assert plain
    assert moved
    for r in plain:
        np.testing.assert_array_equal(r.mask, mask)
        assert r.applied == ["photometric"]
    for r in moved:
        assert r.applied[0] in {"perspective", "edge_crop"}


def _synth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": ["a", "b"],
            "mask_path": ["a", "b"],
            "meta_path": ["a", "b"],
            "ecg_id": [1, 2],
            "patient_id": [10, 20],
            "strat_fold": [1, 10],
            "split": ["train", "test"],
            "variant": [0, 0],
            "seed": [0, 0],
            "params_json": ["{}", "{}"],
        }
    )


def _source() -> pd.DataFrame:
    return _synth_rows()[["ecg_id", "patient_id", "strat_fold", "split"]]


def test_folds_inherited_accepts_matching_manifest() -> None:
    check_folds_inherited(_synth_rows(), _source())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("strat_fold", 10, "changed strat_fold"),
        ("split", "test", "changed split"),
        ("patient_id", 99, "changed patient_id"),
        ("ecg_id", 999, "unknown ecg_id"),
    ],
)
def test_folds_inherited_detects_tampering(column: str, value: int | str, message: str) -> None:
    synth = _synth_rows()
    synth[column] = synth[column].astype(object)
    synth.at[0, column] = value
    with pytest.raises(LeakageError, match=message):
        check_folds_inherited(synth, _source())


def test_patient_split_straddle_is_detected() -> None:
    source = pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "patient_id": [10, 10],
            "strat_fold": [1, 10],
            "split": ["train", "test"],
        }
    )
    synth = _synth_rows()
    synth["patient_id"] = [10, 10]
    with pytest.raises(LeakageError, match="more than one split"):
        check_folds_inherited(synth, source)


def test_synth_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "x" / "synth_manifest.parquet"
    write_synth_manifest(_synth_rows(), path)
    assert list(read_synth_manifest(path).columns) == SYNTH_COLUMNS


def _write_png(path: Path, width: int = 120, height: int = 80) -> None:
    Image.new("RGBA", (width, height), (255, 255, 255, 255)).save(path)


def test_process_image_writes_image_mask_and_meta(tmp_path: Path) -> None:
    png, cfg = tmp_path / "g.png", tmp_path / "g.json"
    _write_png(png)
    cfg.write_text(json.dumps(_config()))
    meta = process_image(
        png,
        cfg,
        tmp_path / "i" / "a.png",
        tmp_path / "m" / "a.png",
        tmp_path / "x" / "a.json",
        None,
    )
    mask = np.asarray(Image.open(tmp_path / "m" / "a.png"))
    assert mask.shape == (80, 120)
    assert mask.max() == RHYTHM_STRIP_ID
    assert np.asarray(Image.open(tmp_path / "i" / "a.png")).shape == (80, 120, 3)
    assert meta["px_per_mm"] == pytest.approx(39.37 / 5)
    assert meta["geometry_modified"] is False
    assert json.loads((tmp_path / "x" / "a.json").read_text()) == meta


def test_process_image_drops_scale_after_geometric_artefacts(tmp_path: Path) -> None:
    png, cfg = tmp_path / "g.png", tmp_path / "g.json"
    _write_png(png)
    cfg.write_text(json.dumps(_config()))
    for seed in range(40):
        meta = process_image(
            png, cfg, tmp_path / "i.png", tmp_path / "m.png", tmp_path / "x.json", seed
        )
        assert (meta["px_per_mm"] is None) == meta["geometry_modified"]
        if meta["geometry_modified"]:
            return
    pytest.fail("no geometric artefact in 40 seeds")


FAKE_GENERATOR = textwrap.dedent(
    """
    import argparse, json
    from pathlib import Path
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    args, _ = parser.parse_known_args()
    for hea in sorted(Path(args.i).glob("*.hea")):
        Image.new("RGBA", (120, 80), (255, 255, 255, 255)).save(Path(args.o) / f"{hea.stem}-0.png")
        config = %s
        (Path(args.o) / f"{hea.stem}-0.json").write_text(json.dumps(config))
    """
)


def test_generate_dataset_inherits_folds_and_cleans_up(tmp_path: Path) -> None:
    kit = tmp_path / "kit"
    script_dir = kit / GENERATOR_SUBDIR
    script_dir.mkdir(parents=True)
    (script_dir / "gen_ecg_images_from_data_batch.py").write_text(FAKE_GENERATOR % repr(_config()))
    root = tmp_path / "ptbxl"
    index = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "strat_fold": [1, 2, 3, 4, 5],
            "filename_lr": [f"records100/{i:05d}_lr" for i in range(1, 6)],
            "filename_hr": [f"records500/{i:05d}_hr" for i in range(1, 6)],
            "age": 50,
            "sex": 0,
        },
        index=pd.Index(range(1, 6), name="ecg_id"),
    )
    (root / "records500").mkdir(parents=True)
    for name in index["filename_hr"]:
        for suffix in (".hea", ".dat"):
            (root / (name + suffix)).write_bytes(b"x")
    records = build_manifest(index)
    out = tmp_path / "out"
    manifest = generate_dataset(
        records,
        root,
        out,
        KitSetup(kit_dir=kit, python=Path(sys.executable)),
        variants=2,
        batch_size=2,
        seed=0,
        artefacts=False,
    )
    assert len(manifest) == 10
    assert set(manifest["variant"]) == {0, 1}
    assert manifest["image_path"].is_unique
    check_folds_inherited(manifest, records)
    for column in ("image_path", "mask_path", "meta_path"):
        assert all((out / p).exists() for p in manifest[column])
    assert not (out / "_work").exists()


def test_select_records_is_reproducible_and_split_specific() -> None:
    frame = pd.DataFrame(
        {"ecg_id": range(20), "split": ["train"] * 15 + ["test"] * 5, "patient_id": range(20)}
    )
    first = select_records(frame, "train", 5, seed=1)
    assert first.equals(select_records(frame, "train", 5, seed=1))
    assert set(first["split"]) == {"train"}
    assert len(select_records(frame, "test", 99, seed=1)) == 5
