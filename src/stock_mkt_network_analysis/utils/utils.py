"""
Utility functions for stock market network analysis.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def align_X_y(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align features and target on common dates, then drop rows with NaNs.
    """
    common = X.index.intersection(y.index)
    X_aligned = X.loc[common].copy()
    y_aligned = y.loc[common].copy()

    valid = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_aligned = X_aligned.loc[valid]
    y_aligned = y_aligned.loc[valid]

    return X_aligned, y_aligned