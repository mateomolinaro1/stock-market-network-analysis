from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


_SCORING_FUNCS: dict[str, Callable] = {
    "roc_auc": safe_roc_auc,
    "pr_auc": safe_average_precision,
}


def get_scoring_func(metric: str) -> Callable:
    if metric not in _SCORING_FUNCS:
        raise ValueError(f"Unknown scoring metric '{metric}'. Choose from: {list(_SCORING_FUNCS)}")
    return _SCORING_FUNCS[metric]


def compute_oos_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute OOS classification metrics per model from a predictions DataFrame.

    Expected columns: y_true, y_pred (probability), y_pred_binary.
    If 'model' is a level in the index, one row is returned per model;
    otherwise a single row indexed as 'all' is returned.

    All metrics are oriented so that higher = better:
      - roc_auc, pr_auc, precision, recall, f1: standard (higher = better)
      - neg_brier_score: negated brier_score_loss, i.e. -E[(y_pred - y_true)^2]
        (original Brier score is lower = better; negation makes it consistent)
    """
    df = predictions.reset_index()
    has_model = "model" in df.columns
    groups = df.groupby("model") if has_model else [("all", df)]

    records = []
    for name, grp in groups:
        valid = grp.dropna(subset=["y_true", "y_pred", "y_pred_binary"])
        if len(valid) == 0:
            records.append({
                "model": name,
                "roc_auc": np.nan, "pr_auc": np.nan,
                "precision": np.nan, "recall": np.nan,
                "f1": np.nan, "neg_brier_score": np.nan,
            })
            continue

        y_true = valid["y_true"].values.astype(int)
        y_pred = valid["y_pred"].values.astype(float)
        y_pred_binary = valid["y_pred_binary"].values.astype(int)

        records.append({
            "model": name,
            "roc_auc": safe_roc_auc(y_true, y_pred),
            "pr_auc": safe_average_precision(y_true, y_pred),
            "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
            "neg_brier_score": float(-brier_score_loss(y_true, y_pred)),
        })

    return pd.DataFrame(records)