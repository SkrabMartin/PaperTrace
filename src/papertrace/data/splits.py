"""Patient-level train/validation/test splits and the leakage check.

PTB-XL ships stratified folds 1-10 that keep all records of a patient in one fold. Folds 1-8
are used for training, fold 9 for validation and fold 10 for the final test.
"""

from typing import Literal

import pandas as pd

Split = Literal["train", "val", "test"]

TRAIN_FOLDS = frozenset(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10


class LeakageError(ValueError):
    """Raised when a patient or record appears in more than one split (or fold)."""


def split_for_fold(fold: int) -> Split:
    """Map a PTB-XL ``strat_fold`` to its split."""
    if fold in TRAIN_FOLDS:
        return "train"
    if fold == VAL_FOLD:
        return "val"
    if fold == TEST_FOLD:
        return "test"
    raise ValueError(f"unknown strat_fold: {fold}")


def assign_splits(index: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``index`` with a ``split`` column derived from ``strat_fold``."""
    result = index.copy()
    result["split"] = result["strat_fold"].map(split_for_fold)
    return result


def check_no_leakage(manifest: pd.DataFrame) -> None:
    """Verify that no patient or record is shared between splits.

    Args:
        manifest: Frame with ``patient_id``, ``strat_fold`` and ``split`` columns and the
            ``ecg_id`` either as index or as a column.

    Raises:
        LeakageError: If a patient is in several splits or folds, or an ``ecg_id`` repeats.
    """
    ecg_ids = manifest["ecg_id"] if "ecg_id" in manifest.columns else manifest.index.to_series()
    if ecg_ids.duplicated().any():
        raise LeakageError("duplicate ecg_id in manifest")
    for column in ("split", "strat_fold"):
        spread = manifest.groupby("patient_id")[column].nunique()
        leaked = spread[spread > 1]
        if not leaked.empty:
            raise LeakageError(
                f"{len(leaked)} patient(s) appear in more than one {column}, "
                f"e.g. patient_id={leaked.index[0]}"
            )
