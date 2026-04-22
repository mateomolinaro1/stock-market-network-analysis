from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RollingCorrelationEstimator:
    lookback: int
    min_non_nan_assets: Optional[int] = None

    def compute_correlation(self, window_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate the correlation matrix from a lookback window.
        """
        clean_window = window_returns.dropna(axis=1, how="any")

        if self.min_non_nan_assets is not None:
            if clean_window.shape[1] < self.min_non_nan_assets:
                return pd.DataFrame()

        if clean_window.shape[1] < 2:
            return pd.DataFrame()

        corr = clean_window.corr()
        corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return corr

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