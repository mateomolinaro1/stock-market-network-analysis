"""
A module for building rolling time-series folds for cross-validation in stock market network analysis.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd


def build_rolling_time_series_folds(
    dates: Sequence[pd.Timestamp],
    train_size: int,
    val_size: int,
    step_size: Optional[int] = None,
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Build rolling time-series folds.

    Example:
        Fold 1: train = [t0 ... t1], val = [t1+1 ... t2]
        Fold 2: train = [t1 ... t2], val = [t2+1 ... t3]
    """
    dates = pd.Index(dates).sort_values()
    step_size = val_size if step_size is None else step_size

    folds = []
    start = 0

    while True:
        train_start = start
        train_end = train_start + train_size
        val_end = train_end + val_size

        if val_end > len(dates):
            break

        train_dates = dates[train_start:train_end]
        val_dates = dates[train_end:val_end]

        folds.append((train_dates, val_dates))
        start += step_size

    return folds
