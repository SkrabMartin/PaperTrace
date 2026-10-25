from pathlib import Path

import pandas as pd
import pytest

from papertrace.data.manifest import build_manifest, read_manifest, write_manifest
from papertrace.data.ptbxl import load_index
from papertrace.data.splits import LeakageError, check_no_leakage, split_for_fold


def _index() -> pd.DataFrame:
    # patients 1..10, one record each, patient n lives in fold n; patient 11 has two records
    rows = [
        {"ecg_id": n, "patient_id": n, "strat_fold": n, "age": 50, "sex": 0} for n in range(1, 11)
    ]
    rows.append({"ecg_id": 11, "patient_id": 11, "strat_fold": 10, "age": 60, "sex": 1})
    rows.append({"ecg_id": 12, "patient_id": 11, "strat_fold": 10, "age": 60, "sex": 1})
    frame = pd.DataFrame(rows).set_index("ecg_id")
    frame["filename_lr"] = [f"records100/{i:05d}_lr" for i in frame.index]
    frame["filename_hr"] = [f"records500/{i:05d}_hr" for i in frame.index]
    return frame


@pytest.mark.parametrize(
    ("fold", "split"),
    [(1, "train"), (8, "train"), (9, "val"), (10, "test")],
)
def test_split_for_fold(fold: int, split: str) -> None:
    assert split_for_fold(fold) == split


@pytest.mark.parametrize("fold", [0, 11, -1])
def test_unknown_fold_is_rejected(fold: int) -> None:
    with pytest.raises(ValueError, match="unknown strat_fold"):
        split_for_fold(fold)


def test_manifest_has_no_patient_leakage() -> None:
    manifest = build_manifest(_index())
    by_split = manifest.groupby("split")["patient_id"].apply(set)
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
    assert by_split["train"].isdisjoint(by_split["val"])
    assert list(manifest["ecg_id"]) == sorted(manifest["ecg_id"])


def test_patient_in_two_splits_is_detected() -> None:
    manifest = build_manifest(_index())
    moved = manifest["ecg_id"] == 12
    manifest.loc[moved, "strat_fold"] = 1
    manifest.loc[moved, "split"] = "train"
    with pytest.raises(LeakageError, match="more than one"):
        check_no_leakage(manifest)


def test_duplicate_ecg_id_is_detected() -> None:
    manifest = build_manifest(_index())
    duplicated = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(LeakageError, match="duplicate ecg_id"):
        check_no_leakage(duplicated)


def test_manifest_round_trips_through_parquet(tmp_path: Path) -> None:
    manifest = build_manifest(_index())
    path = tmp_path / "nested" / "manifest.parquet"
    write_manifest(manifest, path)
    pd.testing.assert_frame_equal(read_manifest(path), manifest)


def test_load_index_reads_ptbxl_columns(tmp_path: Path) -> None:
    (tmp_path / "ptbxl_database.csv").write_text(
        "ecg_id,patient_id,age,sex,strat_fold,filename_lr,filename_hr,report\n"
        "1,15709.0,56.0,1,3,records100/00000/00001_lr,records500/00000/00001_hr,x\n"
    )
    index = load_index(tmp_path)
    assert index.index.name == "ecg_id"
    assert index.loc[1, "patient_id"] == 15709
    assert str(index["patient_id"].dtype) == "int64"
    assert "report" not in index.columns


def test_load_index_rejects_missing_fold(tmp_path: Path) -> None:
    (tmp_path / "ptbxl_database.csv").write_text(
        "ecg_id,patient_id,age,sex,strat_fold,filename_lr,filename_hr\n"
        "1,15709.0,56.0,1,,records100/a,records500/a\n"
    )
    with pytest.raises(ValueError, match="without patient_id or strat_fold"):
        load_index(tmp_path)
