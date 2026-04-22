from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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

    def make_features(
        self,
        returns: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Build rolling features for all eligible dates in returns.
        Feature at date t uses only data before t.
        """
        returns = returns.sort_index()

        corr_cache = self.correlation_estimator.compute_rolling(returns)
        rows = []
        dates = []

        cnt = 0
        for date, corr in corr_cache.items():
            cnt += 1
            logger.info(f"Processing date {date.date()} ({cnt}/{len(list(corr_cache.keys()))}) with correlation matrix of shape {corr.shape}")
            if corr.empty:
                continue

            graph = self.graph_builder.build(corr, threshold)
            feats = self.feature_extractor.transform(graph, corr)

            rows.append(feats)
            dates.append(date)

        return pd.DataFrame(rows, index=pd.Index(dates, name="date"))

    def make_features_for_dates(
        self,
        returns: pd.DataFrame,
        target_dates: Sequence[pd.Timestamp],
        threshold: float,
    ) -> pd.DataFrame:
        """
        Build features only for specific dates, always using past-only data.
        """
        returns = returns.sort_index()
        target_dates = pd.Index(target_dates).sort_values()

        rows = []
        dates = []

        for date in target_dates:
            corr = self.correlation_estimator.compute_for_date(returns, date)
            if corr.empty:
                continue

            graph = self.graph_builder.build(corr, threshold)
            feats = self.feature_extractor.transform(graph, corr)

            rows.append(feats)
            dates.append(date)

        return pd.DataFrame(rows, index=pd.Index(dates, name="date"))