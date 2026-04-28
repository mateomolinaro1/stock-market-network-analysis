"""
TS feature cache: save and load the _ts_feature_cache DataFrame to/from S3 as parquet.

Cache key hashes the parameters that determine the TS features:
  LOOKBACK_CORR (drives the adaptive window sizes), DATA_FREQ, RETURNS_TYPE,
  FORECASTING_HORIZON, and the first/last date of the returns index.

Note: HALFLIFE_CORR is intentionally excluded — TS features do not use
correlation matrices and are therefore independent of it.

S3 layout:
  ts_cache/<cache_key>/ts_features.parquet

Index handling: the DatetimeIndex is reset before upload and restored after load,
consistent with the cv_cache convention.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_S3_PREFIX = "ts_cache"
_FILENAME = "ts_features.parquet"


def compute_ts_cache_key(config, returns: pd.DataFrame) -> str:
    """Return a 16-char hex SHA-256 digest that uniquely identifies a TS feature cache."""
    fields = {
        "LOOKBACK_CORR": config.lookback_corr,
        "DATA_FREQ": config.data_freq,
        "RETURNS_TYPE": config.returns_type,
        "FORECASTING_HORIZON": config.forecasting_horizon,
        "FIRST_DATE": str(returns.index[0].date()),
        "LAST_DATE": str(returns.index[-1].date()),
    }
    canonical = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _s3_key(cache_key: str) -> str:
    return f"{_S3_PREFIX}/{cache_key}/{_FILENAME}"


def save_ts_cache(ts_features: pd.DataFrame, cache_key: str, s3) -> None:
    """Upload the TS feature DataFrame to S3 as parquet."""
    key = _s3_key(cache_key)
    if s3.exists(key):
        logger.warning("Overwriting existing TS cache at %s", key)
    df = ts_features.copy()
    df.index.name = "date"  # force consistent name — from_dict produces index.name=None
    df = df.reset_index()
    s3.upload(df, key, overwrite=True)
    logger.info("Saved TS feature cache to %s", key)


def load_ts_cache(cache_key: str, s3) -> Optional[pd.DataFrame]:
    """Download the TS feature DataFrame from S3. Returns None on cache miss."""
    key = _s3_key(cache_key)
    if not s3.exists(key):
        logger.warning("TS cache miss: %s not found.", key)
        return None
    df = s3.load(key)
    if "date" in df.columns:
        df = df.set_index("date")
    elif "index" in df.columns:
        # Legacy files saved before index.name was forced to "date"
        df = df.set_index("index")
    df.index = pd.to_datetime(df.index)
    logger.info("Loaded TS feature cache from %s (%d rows)", key, len(df))
    return df