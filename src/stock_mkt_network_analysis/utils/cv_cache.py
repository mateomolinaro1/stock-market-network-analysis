"""
CV result caching: save and load WalkForwardResult / NestedCVResult dataframes to/from S3
via the project's better_aws S3 client.

Cache key is a SHA-256 hash of the experiment parameters that fully determine the CV result:
  DATA:        TARGET_VARIABLE, TARGET_VARIABLE_ROLLING_WINDOW, QUANTILE_FOR_DUMMY, RETURNS_TYPE
  FORECASTING: FORECASTING_HORIZON, LOOKBACK_CORR, INNER_TRAIN_SIZE, INNER_VAL_SIZE,
               INNER_STEP_SIZE, THRESHOLD_GRID, SCORING_METRIC, MODEL_GRID

S3 layout:
  cv_results/<cache_key>/predictions.parquet
  cv_results/<cache_key>/selection_history.parquet
  cv_results/<cache_key>/oos_metrics.parquet

Overwrite policy: save_cv_result always overwrites. An explicit WARNING is logged when an
existing file is detected so the overwrite is visible in logs.

Index handling: better_aws uploads DataFrames with index=False. Meaningful indices
(date, model) are reset before upload and re-inferred after load from column names.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_S3_PREFIX = "cv_results"
_FILENAMES = ("predictions", "selection_history", "oos_metrics")


def compute_cache_key(config) -> str:
    """Return a 16-char hex SHA-256 digest of the parameters that determine the CV result."""
    with open(config.RUN_PIPELINE_CONFIG_PATH, "r") as f:
        raw = json.load(f)

    data = raw.get("DATA", {})
    forecasting = raw.get("FORECASTING", {})

    fields = {
        "TARGET_VARIABLE": data.get("TARGET_VARIABLE"),
        "TARGET_VARIABLE_ROLLING_WINDOW": data.get("TARGET_VARIABLE_ROLLING_WINDOW"),
        "QUANTILE_FOR_DUMMY": data.get("QUANTILE_FOR_DUMMY"),
        "RETURNS_TYPE": data.get("RETURNS_TYPE"),
        "FORECASTING_HORIZON": forecasting.get("FORECASTING_HORIZON"),
        "LOOKBACK_CORR": forecasting.get("LOOKBACK_CORR"),
        "INNER_TRAIN_SIZE": forecasting.get("INNER_TRAIN_SIZE"),
        "INNER_VAL_SIZE": forecasting.get("INNER_VAL_SIZE"),
        "INNER_STEP_SIZE": forecasting.get("INNER_STEP_SIZE"),
        "THRESHOLD_GRID": forecasting.get("THRESHOLD_GRID"),
        "SCORING_METRIC": forecasting.get("SCORING_METRIC"),
        "MODEL_GRID": forecasting.get("MODEL_GRID"),
        "HALFLIFE_CORR": forecasting.get("HALFLIFE_CORR"),
        "EXPANDING_OR_ROLLING": forecasting.get("EXPANDING_OR_ROLLING"),
    }

    canonical = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _s3_key(cache_key: str, name: str, feature_mode: str = "all") -> str:
    return f"{_S3_PREFIX}/{cache_key}/{feature_mode}/{name}.parquet"


def _prepare_for_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index before upload so meaningful index columns are not dropped by better_aws."""
    if isinstance(df.index, pd.RangeIndex):
        return df
    return df.reset_index()


def _infer_and_set_index(df: pd.DataFrame) -> pd.DataFrame:
    """Re-set the appropriate index after loading, based on which columns are present."""
    if "date" in df.columns and "model" in df.columns:
        return df.set_index(["date", "model"])
    if "date" in df.columns:
        return df.set_index("date")
    return df


def save_cv_result(result, cache_key: str, s3, feature_mode: str = "all") -> None:
    """
    Upload predictions, selection_history, and oos_metrics to S3 as parquet files.

    WARNING: always overwrites existing files. An explicit WARNING log is emitted
    when a file already exists so the overwrite is visible in logs.
    """
    frames = {
        "predictions": result.predictions,
        "selection_history": result.selection_history,
        "oos_metrics": result.oos_metrics,
    }

    for name, df in frames.items():
        key = _s3_key(cache_key, name, feature_mode)
        if s3.exists(key):
            logger.warning(f"Overwriting existing CV result at {key}")
        s3.upload(_prepare_for_upload(df), key, overwrite=True)
        logger.info(f"Saved {name}.parquet to {key}")


def load_cv_result(cache_key: str, s3, feature_mode: str = "all") -> Optional[dict]:
    """
    Download predictions, selection_history, and oos_metrics from S3.

    Returns a dict with keys matching WalkForwardResult / NestedCVResult fields,
    or None if any file is missing.
    """
    frames = {}
    for name in _FILENAMES:
        key = _s3_key(cache_key, name, feature_mode)
        if not s3.exists(key):
            logger.warning(f"CV cache miss: {key} not found.")
            return None
        df = s3.load(key)
        frames[name] = _infer_and_set_index(df)
        logger.info(f"Loaded {name}.parquet from {key}")
    return frames