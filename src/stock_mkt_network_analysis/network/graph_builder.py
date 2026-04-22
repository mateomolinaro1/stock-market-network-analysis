# systemic_risk_network/network/graph_builder.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx


def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Mantegna distance.
    """
    dist = np.sqrt(2.0 * (1.0 - corr.clip(-1.0, 1.0)))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


@dataclass
class ThresholdGraphBuilder:
    use_absolute_threshold: bool = True
    keep_sign: bool = True

    def threshold_adjacency(self, corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Build weighted adjacency matrix from a correlation matrix.
        """
        if corr.empty:
            return pd.DataFrame()

        adj = corr.copy()

        if self.use_absolute_threshold:
            mask = np.abs(adj.values) >= threshold
        else:
            mask = adj.values >= threshold

        np.fill_diagonal(mask, False)
        adj = adj.where(mask, other=0.0)

        if not self.keep_sign:
            adj = adj.abs()

        arr = adj.to_numpy(copy=True)
        np.fill_diagonal(arr, 0.0)
        return pd.DataFrame(arr, index=adj.index, columns=adj.columns)

    def build(self, corr: pd.DataFrame, threshold: float) -> nx.Graph:
        """
        Build thresholded graph from correlation matrix.
        """
        adj = self.threshold_adjacency(corr, threshold)
        if adj.empty:
            return nx.Graph()
        return nx.from_pandas_adjacency(adj)