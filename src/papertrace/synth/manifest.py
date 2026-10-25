"""The synthetic-image manifest and its leakage guarantees.

Every synthetic image inherits ``patient_id``, ``strat_fold`` and ``split`` from the PTB-XL record
it was drawn from, so images of one patient never straddle two splits.
"""

from pathlib import Path

import pandas as pd

from papertrace.data.splits import LeakageError

SYNTH_COLUMNS = [
    "image_path",
    "mask_path",
    "meta_path",
    "ecg_id",
    "patient_id",
    "strat_fold",
    "split",
    "variant",
    "seed",
    "params_json",
]
_INHERITED = ["patient_id", "strat_fold", "split"]


def check_folds_inherited(synth: pd.DataFrame, split_manifest: pd.DataFrame) -> None:
    """Verify that each synthetic image carries the patient, fold and split of its source record.

    Raises:
        LeakageError: If an ``ecg_id`` is unknown, a column differs, or a patient maps to
            more than one split.
    """
    source = split_manifest.set_index("ecg_id")[_INHERITED]
    unknown = ~synth["ecg_id"].isin(source.index)
    if unknown.any():
        raise LeakageError(f"{int(unknown.sum())} synthetic image(s) have an unknown ecg_id")
    expected = source.loc[synth["ecg_id"]].reset_index(drop=True)
    for column in _INHERITED:
        mismatched = expected[column].to_numpy() != synth[column].to_numpy()
        if mismatched.any():
            raise LeakageError(f"{int(mismatched.sum())} synthetic image(s) changed {column}")
    if (synth.groupby("patient_id")["split"].nunique() > 1).any():
        raise LeakageError("a patient has synthetic images in more than one split")


def write_synth_manifest(manifest: pd.DataFrame, path: Path) -> None:
    """Write the synthetic manifest as Parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest[SYNTH_COLUMNS].to_parquet(path, index=False)


def read_synth_manifest(path: Path) -> pd.DataFrame:
    """Read a synthetic manifest written by :func:`write_synth_manifest`."""
    return pd.read_parquet(path)
