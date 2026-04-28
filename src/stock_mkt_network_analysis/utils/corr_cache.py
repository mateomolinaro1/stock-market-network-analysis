"""
Rolling correlation cache: save and load the full _corr_cache dict
(Dict[pd.Timestamp, pd.DataFrame]) to/from S3 as a gzip-compressed pickle.

Cache key is a SHA-256 hash of the parameters that fully determine the
correlation matrices:
  LOOKBACK_CORR, HALFLIFE_CORR, RETURNS_TYPE, DATA_FREQ,
  and the first/last date of the returns index (data vintage guard).

S3 layout:
  corr_cache/<cache_key>/corr_matrices.pkl.gz

Better-aws does not expose a raw-bytes upload API, so we use boto3 directly.
Credentials are picked up from the environment (same as the rest of the project).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import pickle
from typing import Dict, Optional

import pandas as pd
from botocore.exceptions import ClientError
import boto3

logger = logging.getLogger(__name__)

_S3_PREFIX = "corr_cache"
_FILENAME = "corr_matrices.pkl.gz"


def compute_corr_cache_key(config, returns: pd.DataFrame) -> str:
    """
    Return a 16-char hex SHA-256 digest that uniquely identifies a correlation cache.

    Includes the first and last date of the returns index so the cache is
    automatically invalidated when a new data vintage extends the history.
    """
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


def _s3_key(cache_key: str) -> str:
    return f"{_S3_PREFIX}/{cache_key}/{_FILENAME}"


def _boto3_client(region: str):
    return boto3.client("s3", region_name=region)


def save_corr_cache(
    corr_cache: Dict[pd.Timestamp, pd.DataFrame],
    cache_key: str,
    bucket: str,
    region: str,
) -> None:
    """Serialize corr_cache as gzip-compressed pickle and upload to S3."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
        pickle.dump(corr_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)

    key = _s3_key(cache_key)
    client = _boto3_client(region)
    client.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info("Saved corr cache to s3://%s/%s", bucket, key)


def load_corr_cache(
    cache_key: str,
    bucket: str,
    region: str,
) -> Optional[Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Download and deserialize the corr cache from S3.

    Returns the cache dict, or None on a cache miss.
    """
    key = _s3_key(cache_key)
    client = _boto3_client(region)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read()
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            logger.warning("Corr cache miss: s3://%s/%s not found.", bucket, key)
            return None
        raise

    buf = io.BytesIO(raw)
    with gzip.GzipFile(fileobj=buf, mode="rb") as f:
        cache = pickle.load(f)

    logger.info(
        "Loaded corr cache from s3://%s/%s (%d matrices)", bucket, key, len(cache)
    )
    return cache