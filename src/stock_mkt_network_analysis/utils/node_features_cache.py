"""
Node-level network feature cache: save and load per-threshold DataFrames to/from S3 as parquet.

Each threshold gets its own file. The MultiIndex(date, stock_id) is reset to regular columns
before upload so better_aws preserves it correctly.

Cache key hashes:
  LOOKBACK_CORR, HALFLIFE_CORR, RETURNS_TYPE, DATA_FREQ,
  and the first/last date of the returns index.
  (threshold is NOT in the key — it is encoded in the S3 filename instead)

S3 layout:
  node_features_cache/<cache_key>/threshold_<threshold>.parquet
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_S3_PREFIX = "node_features_cache"


def compute_node_features_cache_key(config, returns: pd.DataFrame) -> str:
    """Return a 16-char hex SHA-256 digest that uniquely identifies a node feature cache."""
    fields = {
        "LOOKBACK_CORR": config.lookback_corr,
        "HALFLIFE_CORR": config.halflife_corr,
        "RETURNS_TYPE": config.returns_type,
        "DATA_FREQ": config.data_freq,
        "FIRST_DATE": str(returns.index[0].date()),
        "LAST_DATE": str(returns.index[-1].date()),
    }
    canonical = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _s3_key(cache_key: str, threshold: float) -> str:
    return f"{_S3_PREFIX}/{cache_key}/threshold_{threshold}.parquet"


def save_node_features_cache(
    node_feature_cache: Dict[float, pd.DataFrame],
    cache_key: str,
    s3,
) -> None:
    """Upload one parquet per threshold. MultiIndex(date, stock_id) is reset to columns."""
    for threshold, df in node_feature_cache.items():
        key = _s3_key(cache_key, threshold)
        if s3.exists(key):
            logger.warning("Overwriting existing node features cache at %s", key)
        flat = df.reset_index()  # date, stock_id become regular columns
        s3.upload(flat, key, overwrite=True)
        logger.info("Saved node features cache to %s (%d rows)", key, len(flat))


def load_node_features_cache(
    cache_key: str,
    threshold_grid: list,
    s3,
) -> Optional[Dict[float, pd.DataFrame]]:
    """
    Download one parquet per threshold and reconstruct Dict[float, DataFrame].

    Returns None if any threshold file is missing (full recompute required).
    """
    result: Dict[float, pd.DataFrame] = {}
    for threshold in threshold_grid:
        key = _s3_key(cache_key, threshold)
        if not s3.exists(key):
            logger.warning("Node features cache miss for threshold=%s: %s not found.", threshold, key)
            return None
        df = s3.load(key)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "stock_id"])
        result[float(threshold)] = df
        logger.info("Loaded node features cache from %s (%d rows)", key, len(df))
    return result