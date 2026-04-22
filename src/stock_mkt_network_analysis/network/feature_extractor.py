from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import networkx as nx


@dataclass
class BasicNetworkFeatureExtractor:
    """
    Basic feature extractor for thresholded correlation networks.
    """

    @staticmethod
    def transform(
        graph: nx.Graph,
        corr: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        density = nx.density(graph) if n_nodes > 1 else np.nan

        unweighted_degrees = np.array([deg for _, deg in graph.degree()], dtype=float) # store it into a dict with dates as keys so that we will plot degree distribution overtime
        avg_degree = float(np.mean(unweighted_degrees)) if len(unweighted_degrees) else 0.0
        std_degree = float(np.std(unweighted_degrees)) if len(unweighted_degrees) else 0.0

        weighted_degrees = np.array(
            [deg for _, deg in graph.degree(weight="weight")],
            dtype=float,
        ) # save for plot dist
        avg_weighted_degree = float(np.mean(weighted_degrees)) if len(weighted_degrees) else 0.0
        std_weighted_degree = float(np.std(weighted_degrees)) if len(weighted_degrees) else 0.0

        if n_edges > 0:
            edge_weights = np.array(
                [d.get("weight", 0.0) for _, _, d in graph.edges(data=True)],
                dtype=float,
            )
            avg_edge_weight = float(np.mean(edge_weights))
            std_edge_weight = float(np.std(edge_weights))
            max_edge_weight = float(np.max(edge_weights))
            clustering = float(nx.average_clustering(graph, weight="weight"))
        else:
            avg_edge_weight = 0.0
            std_edge_weight = 0.0
            max_edge_weight = 0.0
            clustering = 0.0

        if n_nodes > 0 and n_edges > 0:
            largest_cc = max(nx.connected_components(graph), key=len)
            lcc_ratio = len(largest_cc) / n_nodes
        else:
            lcc_ratio = np.nan

        avg_abs_corr = np.nan
        std_abs_corr = np.nan
        eig1 = np.nan
        eig1_ratio = np.nan

        if corr is not None and not corr.empty:
            vals = corr.values
            iu = np.triu_indices_from(vals, k=1)
            off_diag = vals[iu]

            if off_diag.size > 0:
                avg_abs_corr = float(np.mean(np.abs(off_diag)))
                std_abs_corr = float(np.std(np.abs(off_diag)))

            eigvals = np.linalg.eigvalsh(vals)
            eigvals = np.sort(eigvals)[::-1]
            eig1 = float(eigvals[0])
            eig1_ratio = float(eigvals[0] / np.sum(eigvals))

        return {
            "n_nodes": float(n_nodes),
            "n_edges": float(n_edges),
            "density": float(density) if pd.notna(density) else np.nan,
            "avg_degree": avg_degree,
            "std_degree": std_degree,
            "avg_weighted_degree": avg_weighted_degree,
            "std_weighted_degree": std_weighted_degree,
            "avg_edge_weight": avg_edge_weight,
            "std_edge_weight": std_edge_weight,
            "max_edge_weight": max_edge_weight,
            "clustering": clustering,
            "lcc_ratio": float(lcc_ratio) if pd.notna(lcc_ratio) else np.nan,
            "avg_abs_corr": avg_abs_corr,
            "std_abs_corr": std_abs_corr,
            "eig1": eig1,
            "eig1_ratio": eig1_ratio,
        }