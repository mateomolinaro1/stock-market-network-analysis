from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """
    Return ROC AUC, or NaN if only one class is present.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_score))