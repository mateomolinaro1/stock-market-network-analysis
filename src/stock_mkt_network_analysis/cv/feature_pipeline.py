from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import logging
import pandas as pd

from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.feature_extractor import BasicNetworkFeatureExtractor
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder
from stock_mkt_network_analysis.time_series.adaptive_time_series_feature_extractor import (
    AdaptiveTimeSeriesFeatureExtractor,
)

logger = logging.getLogger(__name__)

@dataclass
class RollingNetworkFeaturePipeline:
    correlation_estimator: RollingCorrelationEstimator
    graph_builder: ThresholdGraphBuilder
    feature_extractor: BasicNetworkFeatureExtractor
    time_series_feature_extractor: Optional[AdaptiveTimeSeriesFeatureExtractor] = None
    _corr_cache: Optional[Dict[pd.Timestamp, pd.DataFrame]] = field(
        default=None, init=False, repr=False
    )
    _feature_cache: Optional[Dict[Tuple[pd.Timestamp, float], Dict]] = field(
        default=None, init=False, repr=False
    )
    _ts_feature_cache: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _market_returns: Optional[pd.DataFrame | pd.Series] = field(default=None, init=False, repr=False)
    _risk_free_returns: Optional[pd.DataFrame | pd.Series] = field(default=None, init=False, repr=False)
    _volumes: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def precompute_cache(
        self,
        returns: pd.DataFrame,
        market_returns: Optional[pd.DataFrame | pd.Series] = None,
        risk_free_returns: Optional[pd.DataFrame | pd.Series] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Pre-compute rolling correlation matrices and optional time-series features
        once on the full returns.
        """
        logger.info("Pre-computing rolling correlation cache on full returns...")
        self._corr_cache = self.correlation_estimator.compute_rolling(returns)
        self._feature_cache = {}
        self._market_returns = market_returns
        self._risk_free_returns = risk_free_returns
        self._volumes = volumes

        if self.time_series_feature_extractor is not None:
            logger.info("Pre-computing adaptive time-series feature cache on full returns...")
            self._ts_feature_cache = self.time_series_feature_extractor.compute_rolling(
                asset_returns=returns,
                market_returns=market_returns,
                risk_free_returns=risk_free_returns,
                volumes=volumes,
            )

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

        if self._ts_feature_cache is not None:
            self._ts_feature_cache = self._ts_feature_cache.loc[self._ts_feature_cache.index >= cutoff_date]

    def _get_features(self, date: pd.Timestamp, corr: pd.DataFrame, threshold: float) -> Dict:
        """
        Return network features for (date, threshold), computing and caching on first call.
        """
        if self._feature_cache is not None:
            key = (date, threshold)
            if key not in self._feature_cache:
                graph = self.graph_builder.build(corr, threshold)
                self._feature_cache[key] = self.feature_extractor.transform(graph, corr)
            return self._feature_cache[key]

        graph = self.graph_builder.build(corr, threshold)
        return self.feature_extractor.transform(graph, corr)

    def _combine_with_time_series_features(
        self,
        network_features: pd.DataFrame,
        returns: pd.DataFrame,
        target_dates: Optional[Sequence[pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        if self.time_series_feature_extractor is None or network_features.empty:
            return network_features

        if self._ts_feature_cache is not None:
            ts_features = self._ts_feature_cache.reindex(network_features.index)
        elif target_dates is None:
            ts_features = self.time_series_feature_extractor.compute_rolling(
                asset_returns=returns,
                market_returns=self._market_returns,
                risk_free_returns=self._risk_free_returns,
                volumes=self._volumes,
            ).reindex(network_features.index)
        else:
            rows = {}
            for date in pd.Index(target_dates).sort_values():
                features = self.time_series_feature_extractor.compute_for_date(
                    asset_returns=returns,
                    date=date,
                    market_returns=self._market_returns,
                    risk_free_returns=self._risk_free_returns,
                    volumes=self._volumes,
                )
                if features:
                    rows[pd.Timestamp(date)] = features
            ts_features = pd.DataFrame.from_dict(rows, orient="index").reindex(network_features.index)

        return network_features.add_prefix("net_").join(ts_features.add_prefix("ts_"), how="inner")

    def make_features(
        self,
        returns: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Build rolling features for all eligible dates in returns.
        Uses correlation, network feature, and optional time-series caches.
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

        network_features = pd.DataFrame(rows, index=pd.Index(dates, name="date"))
        return self._combine_with_time_series_features(network_features, returns)

    def make_features_for_dates(
        self,
        returns: pd.DataFrame,
        target_dates: Sequence[pd.Timestamp],
        threshold: float,
    ) -> pd.DataFrame:
        """
        Build features only for specific dates.
        Uses correlation, network feature, and optional time-series caches.
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

        network_features = pd.DataFrame(rows, index=pd.Index(dates, name="date"))
        return self._combine_with_time_series_features(network_features, returns, target_dates)
