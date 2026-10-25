"""Reading the PTB-XL dataset: the metadata index and individual WFDB records."""

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from numpy.typing import NDArray

DATABASE_FILE = "ptbxl_database.csv"
INDEX_COLUMNS = ["patient_id", "strat_fold", "filename_lr", "filename_hr", "age", "sex"]


def load_index(root: Path) -> pd.DataFrame:
    """Load the PTB-XL metadata table, indexed by ``ecg_id``.

    Args:
        root: Directory containing ``ptbxl_database.csv``.

    Returns:
        One row per record with ``INDEX_COLUMNS``; ``patient_id`` and ``strat_fold`` are integers.

    Raises:
        ValueError: If a record has no patient id or fold.
    """
    frame = pd.read_csv(root / DATABASE_FILE, index_col="ecg_id")
    frame = frame[INDEX_COLUMNS]
    if frame[["patient_id", "strat_fold"]].isna().any().any():
        raise ValueError("PTB-XL index has records without patient_id or strat_fold")
    return frame.astype({"patient_id": "int64", "strat_fold": "int64"})


def load_signal(root: Path, filename: str) -> tuple[NDArray[np.float64], list[str]]:
    """Read one record.

    Args:
        root: PTB-XL root directory.
        filename: Record path relative to ``root`` without extension (``filename_hr`` column).

    Returns:
        The signal as ``(samples, leads)`` in millivolts, and the lead names.
    """
    signal, fields = wfdb.rdsamp(str(root / filename))
    return np.asarray(signal, dtype=np.float64), list(fields["sig_name"])
