"""
Utility functions for stock market network analysis.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def align_x_y(
        x: pd.DataFrame,
        y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align features and target on common dates, then drop rows with NaNs.
    """
    common = x.index.intersection(y.index)
    X_aligned = x.loc[common].copy()
    y_aligned = y.loc[common].copy()

    # Fix the boolean indexing - need to flatten y_aligned for multi-column case
    if isinstance(y_aligned, pd.DataFrame):
        y_isna = y_aligned.isna().any(axis=1)
    else:
        y_isna = y_aligned.isna()

    valid = ~(X_aligned.isna().any(axis=1) | y_isna)
    X_aligned = X_aligned.loc[valid]
    y_aligned = y_aligned.loc[valid]

    return X_aligned, y_aligned