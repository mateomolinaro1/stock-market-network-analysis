"""
Graph-level network feature cache: save and load _feature_cache to/from S3 as parquet.

_feature_cache is Dict[(pd.Timestamp, float), Dict[str, float]] — one entry per
(date, threshold) pair. It is serialised as a flat DataFrame where date and threshold
are regular columns (not index), so better_aws preserves them correctly on upload.

Cache key hashes the parameters that determine graph-level features:
  LOOKBACK_CORR, HALFLIFE_CORR, RETURNS_TYPE, DATA_FREQ,
  THRESHOLD_GRID (sorted), and the first/last date of the returns index.

S3 layout:
  graph_features_cache/<cache_key>/graph_features.parquet
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_S3_PREFIX = "graph_features_cache"
_FILENAME = "graph_features.parquet"


def compute_graph_features_cache_key(config, returns: pd.DataFrame) -> str:
    """Return a 16-char hex SHA-256 digest that uniquely identifies a graph feature cache."""
    fields = {
        "LOOKBACK_CORR": config.lookback_corr,
        "HALFLIFE_CORR": config.halflife_corr,
        "RETURNS_TYPE": config.returns_type,
        "DATA_FREQ": config.data_freq,
        "THRESHOLD_GRID": sorted(config.threshold_grid),
        "FIRST_DATE": str(returns.index[0].date()),
        "LAST_DATE": str(returns.index[-1].date()),
    }
    canonical = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _s3_key(cache_key: str) -> str:
    return f"{_S3_PREFIX}/{cache_key}/{_FILENAME}"


def save_graph_features_cache(
    feature_cache: Dict[Tuple[pd.Timestamp, float], Dict[str, float]],
    cache_key: str,
    s3,
) -> None:
    """
    Serialise _feature_cache as a flat DataFrame and upload to S3.

    date and threshold are stored as regular columns (not index) so that
    better_aws's index=False upload preserves them.
    """
    if not feature_cache:
        logger.warning("Graph features cache is empty — nothing to save.")
        return

    rows = []
    for (date, threshold), features in feature_cache.items():
        row = {"date": date, "threshold": float(threshold), **features}
        rows.append(row)
    df = pd.DataFrame(rows)  # flat: date, threshold, net_feature_1, ...

    key = _s3_key(cache_key)
    if s3.exists(key):
        logger.warning("Overwriting existing graph features cache at %s", key)
    s3.upload(df, key, overwrite=True)
    logger.info("Saved graph features cache to %s (%d rows)", key, len(df))


def load_graph_features_cache(
    cache_key: str,
    s3,
) -> Optional[Dict[Tuple[pd.Timestamp, float], Dict[str, float]]]:
    """
    Download the graph features parquet and reconstruct the _feature_cache dict.

    Returns None on cache miss.
    """
    key = _s3_key(cache_key)
    if not s3.exists(key):
        logger.warning("Graph features cache miss: %s not found.", key)
        return None

    df = s3.load(key)
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = [c for c in df.columns if c not in ("date", "threshold")]
    cache: Dict[Tuple[pd.Timestamp, float], Dict[str, float]] = {
        (pd.Timestamp(row.date), float(row.threshold)): {
            col: float(getattr(row, col)) for col in feature_cols
        }
        for row in df.itertuples(index=False)
    }

    logger.info(
        "Loaded graph features cache from %s (%d entries)", key, len(cache)
    )
    return cache