"""The split manifest: one row per PTB-XL record with its split, shared by all later steps."""

from pathlib import Path

import pandas as pd

from papertrace.data.splits import assign_splits, check_no_leakage

MANIFEST_COLUMNS = [
    "ecg_id",
    "patient_id",
    "strat_fold",
    "split",
    "filename_lr",
    "filename_hr",
]


def build_manifest(index: pd.DataFrame) -> pd.DataFrame:
    """Build and validate the split manifest from the PTB-XL index.

    Raises:
        LeakageError: If the resulting splits are not disjoint by patient and record.
    """
    manifest = assign_splits(index).reset_index().sort_values("ecg_id", ignore_index=True)
    check_no_leakage(manifest)
    return manifest[MANIFEST_COLUMNS]


def write_manifest(manifest: pd.DataFrame, path: Path) -> None:
    """Write the manifest as Parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)


def read_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest written by :func:`write_manifest` and re-check it for leakage."""
    manifest = pd.read_parquet(path)
    check_no_leakage(manifest)
    return manifest
