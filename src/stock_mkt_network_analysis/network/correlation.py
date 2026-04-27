from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


@dataclass
class RollingCorrelationEstimator:
    lookback: int
    min_non_nan_assets: Optional[int] = None
    halflife: Optional[int] = None

    def compute_correlation(self, window_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate the correlation matrix from a lookback window.

        If halflife is set, uses exponential weighting + Ledoit-Wolf shrinkage.
        Otherwise falls back to sample correlation via np.corrcoef.
        """
        clean_window = window_returns.dropna(axis=1, how="any")

        if self.min_non_nan_assets is not None:
            if clean_window.shape[1] < self.min_non_nan_assets:
                return pd.DataFrame()

        if clean_window.shape[1] < 2:
            return pd.DataFrame()

        T, N = clean_window.shape
        arr = clean_window.to_numpy(dtype=float)

        if self.halflife is not None:
            decay = 0.5 ** (1.0 / self.halflife)
            # Most-recent observation gets highest weight
            raw_w = decay ** np.arange(T - 1, -1, -1)
            weights = raw_w / raw_w.sum()

            mu_w = (weights[:, None] * arr).sum(axis=0)
            arr_c = arr - mu_w
            # Scale so (1/T) * Z^T Z = S_w (weighted sample covariance)
            z = arr_c * np.sqrt(weights[:, None] * T)

            lw = LedoitWolf(assume_centered=True)
            lw.fit(z)
            cov = lw.covariance_

            std = np.sqrt(np.diag(cov))
            if np.any(std < 1e-12):
                return pd.DataFrame()
            corr_arr = cov / np.outer(std, std)
            np.clip(corr_arr, -1.0, 1.0, out=corr_arr)
        else:
            corr_arr = np.corrcoef(arr.T)

        corr = pd.DataFrame(corr_arr, index=clean_window.columns, columns=clean_window.columns)
        corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return corr.astype("float32")

    def compute_for_date(self, returns: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute correlation using the `lookback` observations strictly before `date`.
        """
        returns = returns.sort_index()

        if date not in returns.index:
            return pd.DataFrame()

        loc = returns.index.get_loc(date)
        if isinstance(loc, slice):
            return pd.DataFrame()

        if loc < self.lookback:
            return pd.DataFrame()

        window = returns.iloc[loc - self.lookback:loc]
        assert window.index.max() < date # Lookback window must be strictly before target date
        return self.compute_correlation(window)

    def compute_rolling(self, returns: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Precompute rolling correlation matrices for all eligible dates.
        """
        returns = returns.sort_index()
        corr_cache: Dict[pd.Timestamp, pd.DataFrame] = {}
        total = len(returns) - self.lookback
        log_every = max(1, total // 10)

        logger.info(f"Starting rolling correlation: {total} dates, lookback={self.lookback}")

        for i in range(self.lookback, len(returns)):
            date = returns.index[i]
            window = returns.iloc[i - self.lookback:i]
            corr_cache[date] = self.compute_correlation(window)

            step = i - self.lookback + 1
            if step % log_every == 0 or step == total:
                logger.info(f"  {step}/{total} ({100 * step // total}%) — last date: {date.date()}")

        logger.info(f"Rolling correlation complete: {len(corr_cache)} matrices computed")
        return corr_cache