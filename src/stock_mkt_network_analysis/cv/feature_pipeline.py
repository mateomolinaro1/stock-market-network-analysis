from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import logging
import pandas as pd

from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.feature_extractor import BasicNetworkFeatureExtractor, NodeLevelNetworkFeatureExtractor
from stock_mkt_network_analysis.utils.corr_cache import load_corr_cache, save_corr_cache
from stock_mkt_network_analysis.utils.ts_cache import load_ts_cache, save_ts_cache
from stock_mkt_network_analysis.utils.graph_features_cache import (
    load_graph_features_cache,
    save_graph_features_cache,
)
from stock_mkt_network_analysis.utils.node_features_cache import (
    load_node_features_cache,
    save_node_features_cache,
)
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
    _node_feature_cache: Optional[Dict[float, pd.DataFrame]] = field(default=None, init=False, repr=False)
    _market_returns: Optional[pd.DataFrame | pd.Series] = field(default=None, init=False, repr=False)
    _risk_free_returns: Optional[pd.DataFrame | pd.Series] = field(default=None, init=False, repr=False)
    _volumes: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def precompute_cache(
        self,
        returns: pd.DataFrame,
        market_returns: Optional[pd.DataFrame | pd.Series] = None,
        risk_free_returns: Optional[pd.DataFrame | pd.Series] = None,
        volumes: Optional[pd.DataFrame] = None,
        load_or_compute_corr: str = "compute",
        save_corr: bool = False,
        corr_cache_key: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_region: Optional[str] = None,
        load_or_compute_ts: str = "compute",
        save_ts: bool = False,
        ts_cache_key: Optional[str] = None,
        load_or_compute_graph_features: str = "compute",
        save_graph_features: bool = False,
        graph_features_cache_key: Optional[str] = None,
        threshold_grid: Optional[Sequence[float]] = None,
        load_or_compute_node_features: str = "compute",
        save_node_features: bool = False,
        node_features_cache_key: Optional[str] = None,
        s3=None,
    ) -> None:
        """
        Pre-compute rolling correlation matrices and optional time-series features
        once on the full returns.

        load_or_compute_corr : "load" tries S3 first (falls back to compute on miss);
                               "compute" always recomputes.
        save_corr            : when True, uploads the computed corr cache to S3.
        corr_cache_key       : pre-computed cache key (from compute_corr_cache_key).
        s3_bucket / s3_region: S3 coordinates for corr cache (boto3 direct upload).
        load_or_compute_ts            : same semantics as load_or_compute_corr but for TS features.
        save_ts                       : when True, uploads the computed TS feature cache to S3.
        ts_cache_key                  : pre-computed cache key (from compute_ts_cache_key).
        load_or_compute_graph_features: "load" tries S3 first; "compute" always recomputes.
        save_graph_features           : when True, uploads the graph feature cache to S3.
        graph_features_cache_key      : pre-computed cache key (from compute_graph_features_cache_key).
        threshold_grid                : list of thresholds to eager-precompute graph/node features for.
        load_or_compute_node_features : same semantics for node-level features.
        save_node_features            : when True, uploads the node feature cache to S3.
        node_features_cache_key       : pre-computed cache key (from compute_node_features_cache_key).
        s3                            : better_aws S3 client instance (for TS/graph/node parquet upload/load).
        """
        _can_cache = corr_cache_key is not None and s3_bucket is not None and s3_region is not None

        if load_or_compute_corr == "load" and _can_cache:
            logger.info("Trying to load corr cache from S3 (key=%s)...", corr_cache_key)
            loaded = load_corr_cache(corr_cache_key, s3_bucket, s3_region)
            if loaded is not None:
                self._corr_cache = loaded
            else:
                logger.warning("Corr cache miss — computing from scratch.")
                self._corr_cache = self.correlation_estimator.compute_rolling(returns)
                if save_corr:
                    save_corr_cache(self._corr_cache, corr_cache_key, s3_bucket, s3_region)
        else:
            logger.info("Pre-computing rolling correlation cache on full returns...")
            self._corr_cache = self.correlation_estimator.compute_rolling(returns)
            if save_corr and _can_cache:
                save_corr_cache(self._corr_cache, corr_cache_key, s3_bucket, s3_region)

        self._feature_cache = {}
        self._market_returns = market_returns
        self._risk_free_returns = risk_free_returns
        self._volumes = volumes

        if self.time_series_feature_extractor is not None:
            _can_ts_cache = ts_cache_key is not None and s3 is not None
            if load_or_compute_ts == "load" and _can_ts_cache:
                logger.info("Trying to load TS feature cache from S3 (key=%s)...", ts_cache_key)
                loaded_ts = load_ts_cache(ts_cache_key, s3)
                if loaded_ts is not None:
                    self._ts_feature_cache = loaded_ts
                else:
                    logger.warning("TS cache miss — computing from scratch.")
                    self._ts_feature_cache = self.time_series_feature_extractor.compute_rolling(
                        asset_returns=returns,
                        market_returns=market_returns,
                        risk_free_returns=risk_free_returns,
                        volumes=volumes,
                    )
                    if save_ts:
                        save_ts_cache(self._ts_feature_cache, ts_cache_key, s3)
            else:
                logger.info("Pre-computing adaptive time-series feature cache on full returns...")
                self._ts_feature_cache = self.time_series_feature_extractor.compute_rolling(
                    asset_returns=returns,
                    market_returns=market_returns,
                    risk_free_returns=risk_free_returns,
                    volumes=volumes,
                )
                if save_ts and _can_ts_cache:
                    save_ts_cache(self._ts_feature_cache, ts_cache_key, s3)

        # ------------------------------------------------------------------
        # Graph-level features — eager precomputation over all (date, threshold)
        # ------------------------------------------------------------------
        _can_graph_cache = graph_features_cache_key is not None and s3 is not None
        _thresholds = list(threshold_grid) if threshold_grid is not None else []

        if load_or_compute_graph_features == "load" and _can_graph_cache:
            logger.info(
                "Trying to load graph features cache from S3 (key=%s)...", graph_features_cache_key
            )
            loaded_gf = load_graph_features_cache(graph_features_cache_key, s3)
            if loaded_gf is not None:
                self._feature_cache = loaded_gf
            else:
                logger.warning("Graph features cache miss — computing from scratch.")
                self._feature_cache = self._eager_compute_graph_features(_thresholds)
                if save_graph_features:
                    save_graph_features_cache(self._feature_cache, graph_features_cache_key, s3)
        elif _thresholds:
            logger.info("Pre-computing graph-level feature cache for %d threshold(s)...", len(_thresholds))
            self._feature_cache = self._eager_compute_graph_features(_thresholds)
            if save_graph_features and _can_graph_cache:
                save_graph_features_cache(self._feature_cache, graph_features_cache_key, s3)

        # ------------------------------------------------------------------
        # Node-level features — eager precomputation per threshold
        # ------------------------------------------------------------------
        _can_node_cache = node_features_cache_key is not None and s3 is not None

        if load_or_compute_node_features == "load" and _can_node_cache:
            logger.info(
                "Trying to load node features cache from S3 (key=%s)...", node_features_cache_key
            )
            loaded_nf = load_node_features_cache(node_features_cache_key, _thresholds, s3)
            if loaded_nf is not None:
                self._node_feature_cache = loaded_nf
            else:
                logger.warning("Node features cache miss — computing from scratch.")
                self._node_feature_cache = self._eager_compute_node_features(_thresholds)
                if save_node_features:
                    save_node_features_cache(self._node_feature_cache, node_features_cache_key, s3)
        elif _thresholds:
            logger.info("Pre-computing node-level feature cache for %d threshold(s)...", len(_thresholds))
            self._node_feature_cache = self._eager_compute_node_features(_thresholds)
            if save_node_features and _can_node_cache:
                save_node_features_cache(self._node_feature_cache, node_features_cache_key, s3)

        logger.info(f"Cache ready: {len(self._corr_cache)} correlation matrices stored")

    def _eager_compute_graph_features(
        self, threshold_grid: Sequence[float]
    ) -> Dict[Tuple[pd.Timestamp, float], Dict]:
        """Compute and return graph-level features for every (date, threshold) pair."""
        from tqdm.auto import tqdm

        cache: Dict[Tuple[pd.Timestamp, float], Dict] = {}
        corr_items = sorted(self._corr_cache.items())
        pairs = [(d, corr, t) for d, corr in corr_items for t in threshold_grid]
        for date, corr, threshold in tqdm(pairs, desc="Graph features", unit="pair"):
            if corr.empty:
                continue
            key = (date, threshold)
            graph = self.graph_builder.build(corr, threshold)
            cache[key] = self.feature_extractor.transform(graph, corr)
        return cache

    def _eager_compute_node_features(
        self, threshold_grid: Sequence[float]
    ) -> Dict[float, pd.DataFrame]:
        """Compute and return node-level features for every threshold."""
        from tqdm.auto import tqdm

        node_extractor = NodeLevelNetworkFeatureExtractor()
        result: Dict[float, pd.DataFrame] = {}
        for threshold in tqdm(threshold_grid, desc="Node features", unit="threshold"):
            frames = []
            for date, corr in sorted(self._corr_cache.items()):
                if corr.empty:
                    continue
                graph = self.graph_builder.build(corr, threshold)
                node_df = node_extractor.transform(graph)
                if node_df.empty:
                    continue
                node_df.index.name = "stock_id"
                node_df["date"] = date
                frames.append(node_df.reset_index())
            if frames:
                df = pd.concat(frames, ignore_index=True).set_index(["date", "stock_id"])
                result[float(threshold)] = df
        return result

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

    @staticmethod
    def _filter_by_mode(features: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode == "all" or features.empty:
            return features
        if mode == "ts":
            return features[[c for c in features.columns if c.startswith("ts_")]]
        if mode == "network":
            return features[[c for c in features.columns if c.startswith("net_")]]
        raise ValueError(f"Unknown feature mode: {mode!r}. Expected 'all', 'ts', or 'network'.")

    def make_features(
        self,
        returns: pd.DataFrame,
        threshold: float,
        mode: str = "all",
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
        combined = self._combine_with_time_series_features(network_features, returns)
        return self._filter_by_mode(combined, mode)

    def compute_node_features(self, returns: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Compute per-node (stock-level) network metrics for every date in the correlation cache.

        Reuses _corr_cache when available so correlation matrices are not recomputed.
        Graphs are rebuilt from cached correlations (cheap: threshold mask only).

        Returns a DataFrame with MultiIndex(date, stock_id) and one column per metric.
        """
        node_extractor = NodeLevelNetworkFeatureExtractor()

        if self._corr_cache is not None:
            corr_items = sorted(self._corr_cache.items())
        else:
            corr_items = sorted(self.correlation_estimator.compute_rolling(returns).items())

        frames = []
        for date, corr in corr_items:
            if corr.empty:
                continue
            graph = self.graph_builder.build(corr, threshold)
            node_df = node_extractor.transform(graph)
            if node_df.empty:
                continue
            node_df.index.name = "stock_id"
            node_df["date"] = date
            frames.append(node_df.reset_index())

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return result.set_index(["date", "stock_id"])

    def make_features_for_dates(
        self,
        returns: pd.DataFrame,
        target_dates: Sequence[pd.Timestamp],
        threshold: float,
        mode: str = "all",
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
        combined = self._combine_with_time_series_features(network_features, returns, target_dates)
        return self._filter_by_mode(combined, mode)
