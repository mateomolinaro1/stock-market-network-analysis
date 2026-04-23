from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import logging

from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder
from stock_mkt_network_analysis.network.feature_extractor import BasicNetworkFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class RollingNetworkFeaturePipeline:
    correlation_estimator: RollingCorrelationEstimator
    graph_builder: ThresholdGraphBuilder
    feature_extractor: BasicNetworkFeatureExtractor
    _corr_cache: Optional[Dict[pd.Timestamp, pd.DataFrame]] = field(
        default=None, init=False, repr=False
    )
    _feature_cache: Optional[Dict[Tuple[pd.Timestamp, float], Dict]] = field(
        default=None, init=False, repr=False
    )

    def precompute_cache(self, returns: pd.DataFrame) -> None:
        """
        Pre-compute rolling correlation matrices once on the full returns.
        Also initialises the feature cache so subsequent calls to make_features
        and make_features_for_dates store and reuse computed features.
        """
        logger.info("Pre-computing rolling correlation cache on full returns...")
        self._corr_cache = self.correlation_estimator.compute_rolling(returns)
        self._feature_cache = {}
        logger.info(f"Cache ready: {len(self._corr_cache)} correlation matrices stored")

    def evict_before(self, cutoff_date: pd.Timestamp) -> None:
        """
        Free cached entries for all dates strictly before cutoff_date.
        Call after each outer CV step once those dates leave the rolling window.
        """
        if self._corr_cache is not None:
            for d in [d for d in self._corr_cache if d < cutoff_date]:
                del self._corr_cache[d]

        if self._feature_cache is not None:
            for k in [k for k in self._feature_cache if k[0] < cutoff_date]:
                del self._feature_cache[k]

    def _get_features(self, date: pd.Timestamp, corr: pd.DataFrame, threshold: float) -> Dict:
        """
        Return features for (date, threshold), computing and caching on first call.
        """
        if self._feature_cache is not None:
            key = (date, threshold)
            if key not in self._feature_cache:
                graph = self.graph_builder.build(corr, threshold)
                self._feature_cache[key] = self.feature_extractor.transform(graph, corr)
            return self._feature_cache[key]

        graph = self.graph_builder.build(corr, threshold)
        return self.feature_extractor.transform(graph, corr)

    def make_features(
        self,
        returns: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Build rolling features for all eligible dates in returns.
        Uses correlation cache and feature cache if available.
        """
        returns = returns.sort_index()

        if self._corr_cache is not None:
            corr_items = [(d, self._corr_cache[d]) for d in returns.index if d in self._corr_cache]
        else:
            corr_items = list(self.correlation_estimator.compute_rolling(returns).items())

        rows = []
        dates = []

        for date, corr in corr_items:
            if corr.empty:
                continue
            rows.append(self._get_features(date, corr, threshold))
            dates.append(date)

        return pd.DataFrame(rows, index=pd.Index(dates, name="date"))

    def make_features_for_dates(
        self,
        returns: pd.DataFrame,
        target_dates: Sequence[pd.Timestamp],
        threshold: float,
    ) -> pd.DataFrame:
        """
        Build features only for specific dates.
        Uses correlation cache and feature cache if available.
        """
        returns = returns.sort_index()
        target_dates = pd.Index(target_dates).sort_values()

        rows = []
        dates = []

        for date in target_dates:
            if self._corr_cache is not None:
                corr = self._corr_cache.get(date, pd.DataFrame())
            else:
                corr = self.correlation_estimator.compute_for_date(returns, date)

            if corr.empty:
                continue

            rows.append(self._get_features(date, corr, threshold))
            dates.append(date)

        return pd.DataFrame(rows, index=pd.Index(dates, name="date"))