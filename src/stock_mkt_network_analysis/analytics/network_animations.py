from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import logging
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: renders to file without GUI event loop

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder

logger = logging.getLogger(__name__)

DEFAULT_ANIMATION_MAX_FRAMES = 180
DEFAULT_ANIMATION_FPS = 8
DEFAULT_ANIMATION_INTERVAL = 125


@dataclass
class RollingNetworkAnimator:
    """
    Build animated network diagnostics from rolling return-based graphs.

    If corr_cache is provided (a pre-computed {date: corr_matrix} dict, e.g. from
    RollingNetworkFeaturePipeline._corr_cache), build_graphs reuses it instead of
    recomputing rolling correlations — the dominant cost when animating.
    """

    correlation_estimator: RollingCorrelationEstimator
    graph_builder: ThresholdGraphBuilder
    threshold: float
    figures_dir: Path | str
    corr_cache: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    _graph_cache: dict[tuple[Optional[str], Optional[str], Optional[int]], Dict[pd.Timestamp, nx.Graph]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.figures_dir = Path(self.figures_dir)

    def build_graphs(
        self,
        returns: pd.DataFrame,
        start_date: Optional[pd.Timestamp | str] = None,
        end_date: Optional[pd.Timestamp | str] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[pd.Timestamp, nx.Graph]:
        """
        Build rolling threshold graphs for the selected period.

        Dates are the eligible target dates produced by the rolling correlation
        estimator, so each graph at date t uses observations strictly before t.
        """
        cache_key = (
            None if start_date is None else str(pd.Timestamp(start_date)),
            None if end_date is None else str(pd.Timestamp(end_date)),
            max_frames,
        )
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]

        returns = returns.sort_index()
        if self.corr_cache is None:
            self.corr_cache = self.correlation_estimator.compute_rolling(returns)
        corr_cache = self.corr_cache
        dates = self._select_dates(corr_cache.keys(), start_date, end_date, max_frames)

        graphs: Dict[pd.Timestamp, nx.Graph] = {}
        for date in dates:
            corr = corr_cache[date]
            if corr.empty:
                continue
            graphs[pd.Timestamp(date)] = self.graph_builder.build(corr, self.threshold)

        if not graphs:
            raise ValueError("No graph could be built for the selected period.")

        self._graph_cache[cache_key] = graphs
        return graphs

    def animate_degree_distribution(
        self,
        returns: pd.DataFrame,
        start_date: Optional[pd.Timestamp | str] = None,
        end_date: Optional[pd.Timestamp | str] = None,
        filename: str = "degree_distribution_over_time.gif",
        fps: int = DEFAULT_ANIMATION_FPS,
        interval: int = DEFAULT_ANIMATION_INTERVAL,
        max_frames: Optional[int] = DEFAULT_ANIMATION_MAX_FRAMES,
        normalize_counts: bool = False,
        xscale: str = "linear",
        yscale: str = "linear",
        y_max_quantile: Optional[float] = None,
        plot_kind: str = "hist",
        target: Optional[pd.Series | pd.DataFrame] = None,
    ) -> Path:
        """
        Animate the unweighted degree distribution through time.
        """
        graphs = self.build_graphs(returns, start_date, end_date, max_frames)
        dates = list(graphs)
        regimes = self._target_regimes(target, dates)
        max_degree = max((max(dict(graph.degree()).values(), default=0) for graph in graphs.values()), default=0)
        bins = self._degree_bins(max_degree, xscale)
        y_limits = self._degree_distribution_y_limits(
            graphs=graphs,
            bins=bins,
            normalize_counts=normalize_counts,
            y_max_quantile=y_max_quantile,
            yscale=yscale,
            plot_kind=plot_kind,
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        def update(frame_idx: int) -> None:
            date = dates[frame_idx]
            graph = graphs[date]
            degrees = [degree for _, degree in graph.degree()]

            ax.clear()
            if plot_kind == "hist":
                ax.hist(degrees, bins=bins, density=normalize_counts, color="#2a9d8f", edgecolor="white")
                ylabel = "Density" if normalize_counts else "Number of nodes"
            elif plot_kind == "pmf":
                k_values, probabilities = self._degree_pmf(degrees)
                ax.scatter(k_values, probabilities, color="#e76f00", s=36)
                ax.plot(k_values, probabilities, color="#e76f00", alpha=0.35)
                ylabel = "P(k)"
            else:
                raise ValueError("plot_kind must be either 'hist' or 'pmf'.")

            self._apply_degree_axis_scaling(ax, max_degree, xscale, yscale, y_limits)
            ax.set_title(f"Degree distribution - {date:%Y-%m-%d}")
            ax.set_xlabel("Degree")
            ax.set_ylabel(ylabel)
            self._annotate_regime(ax, regimes.get(date))
            ax.grid(axis="y", alpha=0.3)

        animation = FuncAnimation(fig, update, frames=len(dates), interval=interval, repeat=True)
        path = self._save_animation(animation, fig, filename, fps)
        logger.info("Degree distribution animation saved to %s", path)
        return path

    def animate_rich_club(
        self,
        returns: pd.DataFrame,
        start_date: Optional[pd.Timestamp | str] = None,
        end_date: Optional[pd.Timestamp | str] = None,
        degree_threshold: Optional[int] = None,
        filename: str = "rich_club_over_time.gif",
        fps: int = DEFAULT_ANIMATION_FPS,
        interval: int = DEFAULT_ANIMATION_INTERVAL,
        max_frames: Optional[int] = DEFAULT_ANIMATION_MAX_FRAMES,
        normalized: bool = False,
        n_random_reference: int = 10,
        random_swaps_per_edge: int = 5,
        random_seed: Optional[int] = 42,
        xscale: str = "linear",
        yscale: str = "linear",
        target: Optional[pd.Series | pd.DataFrame] = None,
    ) -> Path:
        """
        Animate the rich-club coefficient curve phi(k) through time.

        phi(k) is the density of the subgraph induced by nodes with degree > k.
        If normalized=True, phi(k) is divided by the mean phi(k) of random
        degree-preserving reference graphs.
        """
        graphs = self.build_graphs(returns, start_date, end_date, max_frames)
        dates = list(graphs)
        regimes = self._target_regimes(target, dates)
        curves = {
            date: self._rich_club_curve(
                graph=graph,
                normalized=normalized,
                n_random_reference=n_random_reference,
                random_swaps_per_edge=random_swaps_per_edge,
                random_seed=None if random_seed is None else random_seed + idx,
            )
            for idx, (date, graph) in enumerate(graphs.items())
        }
        max_k = max((max(curve.keys(), default=0) for curve in curves.values()), default=0)
        y_limits = self._rich_club_ylim(curves, normalized, yscale)

        fig, ax = plt.subplots(figsize=(10, 6))

        def update(frame_idx: int) -> None:
            date = dates[frame_idx]
            curve = curves[date]
            x = list(range(max_k + 1))
            y = [curve.get(k, np.nan) for k in x]
            if yscale == "log":
                y = [value if np.isfinite(value) and value > 0 else np.nan for value in y]

            ax.clear()
            ax.plot(x, y, marker="o", color="#264653")
            if degree_threshold is not None:
                ax.axvline(degree_threshold, linestyle="--", color="#e76f51", label=f"k = {degree_threshold}")
                selected_phi = curve.get(degree_threshold, np.nan)
                if np.isfinite(selected_phi):
                    ax.scatter([degree_threshold], [selected_phi], color="#e76f51", zorder=3)
                ax.legend()

            self._apply_rich_club_axis_scaling(ax, max_k, y_limits, xscale, yscale)
            title = "Normalized rich-club coefficient" if normalized else "Rich-club coefficient"
            ax.set_title(f"{title} - {date:%Y-%m-%d}")
            ax.set_xlabel("Degree threshold k")
            ax.set_ylabel("phi(k) / phi_random(k)" if normalized else "phi(k)")
            self._annotate_regime(ax, regimes.get(date))
            ax.grid(alpha=0.3)

        animation = FuncAnimation(fig, update, frames=len(dates), interval=interval, repeat=True)
        path = self._save_animation(animation, fig, filename, fps)
        logger.info("Rich-club animation saved to %s", path)
        return path

    def plot_degree_distribution_by_regime(
        self,
        returns: pd.DataFrame,
        target: pd.Series | pd.DataFrame,
        start_date: Optional[pd.Timestamp | str] = None,
        end_date: Optional[pd.Timestamp | str] = None,
        filename: str = "degree_distribution_by_regime.png",
        max_dates: Optional[int] = None,
        normalize_counts: bool = True,
        xscale: str = "log",
        yscale: str = "log",
        plot_kind: str = "pmf",
    ) -> Path:
        """
        Plot average degree distributions by target regime over the selected period.
        """
        graphs = self.build_graphs(returns, start_date, end_date, max_dates)
        regime_graphs = self._split_graphs_by_regime(graphs, target)
        max_degree = max((max(dict(graph.degree()).values(), default=0) for graph in graphs.values()), default=0)
        bins = self._degree_bins(max_degree, xscale)

        fig, ax = plt.subplots(figsize=(10, 6))
        for regime_value, label, color in self._regime_specs():
            graph_list = regime_graphs.get(regime_value, [])
            if not graph_list:
                continue

            if plot_kind == "hist":
                curves = []
                for graph in graph_list:
                    degrees = [degree for _, degree in graph.degree()]
                    counts, edges = np.histogram(degrees, bins=bins, density=normalize_counts)
                    centers = (edges[:-1] + edges[1:]) / 2
                    curves.append(dict(zip(centers, counts)))
                x, y = self._mean_curve(curves)
                ylabel = "Density" if normalize_counts else "Number of nodes"
            elif plot_kind == "pmf":
                curves = []
                for graph in graph_list:
                    degrees = [degree for _, degree in graph.degree()]
                    k_values, probabilities = self._degree_pmf(degrees)
                    curves.append(dict(zip(k_values, probabilities)))
                x, y = self._mean_curve(curves)
                ylabel = "P(k)"
            else:
                raise ValueError("plot_kind must be either 'hist' or 'pmf'.")

            if yscale == "log":
                y = np.where(y > 0, y, np.nan)
            ax.plot(x, y, marker="o", label=label, color=color)

        ax.set_title("Average degree distribution by regime")
        ax.set_xlabel("Degree")
        ax.set_ylabel(ylabel)
        self._apply_static_axis_scaling(ax, xscale, yscale)
        ax.legend()
        ax.grid(alpha=0.3)
        return self._save_figure(fig, filename)

    def plot_rich_club_by_regime(
        self,
        returns: pd.DataFrame,
        target: pd.Series | pd.DataFrame,
        start_date: Optional[pd.Timestamp | str] = None,
        end_date: Optional[pd.Timestamp | str] = None,
        filename: str = "rich_club_by_regime.png",
        max_dates: Optional[int] = None,
        normalized: bool = False,
        n_random_reference: int = 10,
        random_swaps_per_edge: int = 5,
        random_seed: Optional[int] = 42,
        xscale: str = "log",
        yscale: str = "linear",
    ) -> Path:
        """
        Plot average rich-club curves by target regime over the selected period.
        """
        graphs = self.build_graphs(returns, start_date, end_date, max_dates)
        regime_graphs = self._split_graphs_by_regime(graphs, target)

        fig, ax = plt.subplots(figsize=(10, 6))
        for regime_value, label, color in self._regime_specs():
            graph_list = regime_graphs.get(regime_value, [])
            if not graph_list:
                continue

            curves = [
                self._rich_club_curve(
                    graph=graph,
                    normalized=normalized,
                    n_random_reference=n_random_reference,
                    random_swaps_per_edge=random_swaps_per_edge,
                    random_seed=None if random_seed is None else random_seed + idx,
                )
                for idx, graph in enumerate(graph_list)
            ]
            x, y = self._mean_curve(curves)
            if yscale == "log":
                y = np.where(y > 0, y, np.nan)
            ax.plot(x, y, marker="o", label=label, color=color)

        title = "Average normalized rich-club by regime" if normalized else "Average rich-club by regime"
        ax.set_title(title)
        ax.set_xlabel("Degree threshold k")
        ax.set_ylabel("phi(k) / phi_random(k)" if normalized else "phi(k)")
        self._apply_static_axis_scaling(ax, xscale, yscale)
        ax.legend()
        ax.grid(alpha=0.3)
        return self._save_figure(fig, filename)

    @staticmethod
    def _select_dates(
        dates: Iterable[pd.Timestamp],
        start_date: Optional[pd.Timestamp | str],
        end_date: Optional[pd.Timestamp | str],
        max_frames: Optional[int],
    ) -> list[pd.Timestamp]:
        selected = pd.Index(pd.to_datetime(list(dates))).sort_values()

        if start_date is not None:
            selected = selected[selected >= pd.Timestamp(start_date)]
        if end_date is not None:
            selected = selected[selected <= pd.Timestamp(end_date)]

        if max_frames is not None and max_frames > 0 and len(selected) > max_frames:
            positions = np.linspace(0, len(selected) - 1, max_frames).round().astype(int)
            selected = selected.take(np.unique(positions))

        return [pd.Timestamp(date) for date in selected]

    @staticmethod
    def _target_regimes(
        target: Optional[pd.Series | pd.DataFrame],
        dates: Iterable[pd.Timestamp],
    ) -> dict[pd.Timestamp, Optional[int]]:
        if target is None:
            return {}

        target_series = RollingNetworkAnimator._as_target_series(target)
        aligned = target_series.reindex(pd.Index(dates)).dropna()
        return {pd.Timestamp(date): int(value) for date, value in aligned.items()}

    @staticmethod
    def _as_target_series(target: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(target, pd.DataFrame):
            if target.empty:
                return pd.Series(dtype=float)
            target_series = target.iloc[:, 0]
        else:
            target_series = target

        target_series = target_series.copy()
        target_series.index = pd.to_datetime(target_series.index)
        return target_series.sort_index()

    @staticmethod
    def _regime_specs() -> list[tuple[int, str, str]]:
        return [
            (0, "Régime normal", "#2a9d8f"),
            (1, "Régime tendu", "#e76f51"),
        ]

    @staticmethod
    def _regime_label(value: Optional[int]) -> str:
        if value == 0:
            return "Régime normal"
        if value == 1:
            return "Régime tendu"
        return "Régime inconnu"

    @staticmethod
    def _regime_color(value: Optional[int]) -> str:
        if value == 0:
            return "#2a9d8f"
        if value == 1:
            return "#e76f51"
        return "#6c757d"

    @staticmethod
    def _annotate_regime(ax: plt.Axes, regime_value: Optional[int]) -> None:
        if regime_value is None:
            return

        ax.text(
            0.98,
            0.94,
            RollingNetworkAnimator._regime_label(regime_value),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="white",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": RollingNetworkAnimator._regime_color(regime_value),
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )

    @staticmethod
    def _split_graphs_by_regime(
        graphs: Dict[pd.Timestamp, nx.Graph],
        target: pd.Series | pd.DataFrame,
    ) -> dict[int, list[nx.Graph]]:
        regimes = RollingNetworkAnimator._target_regimes(target, graphs.keys())
        grouped: dict[int, list[nx.Graph]] = {0: [], 1: []}
        for date, graph in graphs.items():
            regime_value = regimes.get(date)
            if regime_value in grouped:
                grouped[regime_value].append(graph)
        return grouped

    @staticmethod
    def _mean_curve(curves: list[dict[float | int, float]]) -> tuple[np.ndarray, np.ndarray]:
        keys = sorted({key for curve in curves for key in curve})
        x = np.array(keys, dtype=float)
        y = []
        for key in keys:
            values = np.array([curve.get(key, np.nan) for curve in curves], dtype=float)
            valid_values = values[np.isfinite(values)]
            y.append(np.mean(valid_values) if valid_values.size else np.nan)
        return x, np.array(y, dtype=float)

    @staticmethod
    def _rich_club_curve(
        graph: nx.Graph,
        normalized: bool = False,
        n_random_reference: int = 10,
        random_swaps_per_edge: int = 5,
        random_seed: Optional[int] = 42,
    ) -> dict[int, float]:
        raw_curve = RollingNetworkAnimator._raw_rich_club_curve(graph)

        if not normalized:
            return raw_curve

        if n_random_reference <= 0:
            raise ValueError("n_random_reference must be strictly positive when normalized=True.")

        reference_curves = []
        for idx in range(n_random_reference):
            reference = RollingNetworkAnimator._random_reference_graph(
                graph,
                swaps_per_edge=random_swaps_per_edge,
                seed=None if random_seed is None else random_seed + idx,
            )
            reference_curves.append(RollingNetworkAnimator._raw_rich_club_curve(reference))

        normalized_curve: dict[int, float] = {}
        for k, phi in raw_curve.items():
            random_values = np.array([curve.get(k, np.nan) for curve in reference_curves], dtype=float)
            valid_random_values = random_values[np.isfinite(random_values)]
            random_mean = np.mean(valid_random_values) if valid_random_values.size else np.nan
            if not np.isfinite(phi) or not np.isfinite(random_mean) or random_mean == 0.0:
                normalized_curve[k] = np.nan
            else:
                normalized_curve[k] = phi / random_mean

        return normalized_curve

    @staticmethod
    def _raw_rich_club_curve(graph: nx.Graph) -> dict[int, float]:
        degrees = dict(graph.degree())
        max_degree = max(degrees.values(), default=0)
        curve: dict[int, float] = {}

        for k in range(max_degree + 1):
            rich_nodes = [node for node, degree in degrees.items() if degree > k]
            n_rich = len(rich_nodes)
            if n_rich < 2:
                curve[k] = np.nan
                continue

            subgraph = graph.subgraph(rich_nodes)
            possible_edges = n_rich * (n_rich - 1) / 2
            curve[k] = subgraph.number_of_edges() / possible_edges

        return curve

    @staticmethod
    def _random_reference_graph(graph: nx.Graph, swaps_per_edge: int, seed: Optional[int]) -> nx.Graph:
        reference = nx.Graph()
        reference.add_nodes_from(graph.nodes())
        reference.add_edges_from(graph.edges())

        n_edges = reference.number_of_edges()
        if n_edges < 2 or swaps_per_edge <= 0:
            return reference

        nswap = max(1, swaps_per_edge * n_edges)
        max_tries = max(nswap * 10, 100)
        try:
            nx.double_edge_swap(reference, nswap=nswap, max_tries=max_tries, seed=seed)
        except nx.NetworkXError as exc:
            logger.debug("Could not fully randomize reference graph: %s", exc)
        return reference

    @staticmethod
    def _degree_bins(max_degree: int, xscale: str) -> np.ndarray:
        if xscale == "linear":
            return np.arange(-0.5, max_degree + 1.5, 1.0)
        if xscale == "log":
            positive_max = max(1, max_degree)
            bins = np.unique(np.floor(np.geomspace(1, positive_max + 1, min(20, positive_max + 1))).astype(int))
            return np.unique(np.concatenate(([-0.5, 0.5], bins[bins > 1] - 0.5, [positive_max + 0.5])))
        raise ValueError("xscale must be either 'linear' or 'log'.")

    @staticmethod
    def _degree_distribution_y_limits(
        graphs: Dict[pd.Timestamp, nx.Graph],
        bins: np.ndarray,
        normalize_counts: bool,
        y_max_quantile: Optional[float],
        yscale: str,
        plot_kind: str,
    ) -> tuple[float, float]:
        maxima = []
        positive_values = []
        for graph in graphs.values():
            degrees = [degree for _, degree in graph.degree()]
            if not degrees:
                continue

            if plot_kind == "hist":
                counts, _ = np.histogram(degrees, bins=bins, density=normalize_counts)
                counts = counts[np.isfinite(counts)]
            elif plot_kind == "pmf":
                _, counts = np.unique(degrees, return_counts=True)
                counts = counts.astype(float) / len(degrees)
            else:
                raise ValueError("plot_kind must be either 'hist' or 'pmf'.")

            if counts.size:
                maxima.append(float(counts.max()))
                positive_values.extend(counts[counts > 0].astype(float))

        if not maxima:
            return (0.8, 1.2) if yscale == "log" else (0.0, 1.0)

        if y_max_quantile is not None:
            if not 0 < y_max_quantile <= 1:
                raise ValueError("y_max_quantile must be in (0, 1].")
            ymax = float(np.quantile(maxima, y_max_quantile))
        else:
            ymax = max(maxima)

        if yscale == "log":
            if not positive_values:
                return 0.8, 1.2
            ymin = max(min(positive_values) * 0.8, 1e-8)
            return ymin, max(ymax * 1.2, ymin * 10.0)

        if yscale != "linear":
            raise ValueError("yscale must be either 'linear' or 'log'.")

        return 0.0, max(ymax * 1.1, 1.0)

    @staticmethod
    def _degree_pmf(degrees: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if not degrees:
            return np.array([]), np.array([])

        values, counts = np.unique(degrees, return_counts=True)
        probabilities = counts.astype(float) / len(degrees)
        return values, probabilities

    @staticmethod
    def _apply_degree_axis_scaling(
        ax: plt.Axes,
        max_degree: int,
        xscale: str,
        yscale: str,
        y_limits: tuple[float, float],
    ) -> None:
        if xscale == "log":
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(-0.5, max(1.5, max_degree + 0.5))
        elif xscale == "linear":
            ax.set_xlim(-0.5, max(1.5, max_degree + 0.5))
        else:
            raise ValueError("xscale must be either 'linear' or 'log'.")

        if yscale == "log":
            ax.set_yscale("log")
            ax.set_ylim(*y_limits)
        elif yscale == "linear":
            ax.set_ylim(*y_limits)
        else:
            raise ValueError("yscale must be either 'linear' or 'log'.")

    @staticmethod
    def _rich_club_ylim(
        curves: Dict[pd.Timestamp, dict[int, float]],
        normalized: bool,
        yscale: str,
    ) -> tuple[float, float]:
        values = np.array(
            [value for curve in curves.values() for value in curve.values() if np.isfinite(value)],
            dtype=float,
        )

        if values.size == 0:
            return (0.8, 1.2) if yscale == "log" else (0.0, 1.05)

        if yscale == "log":
            positive = values[values > 0]
            if positive.size == 0:
                return 0.8, 1.2
            return max(float(positive.min()) * 0.9, 1e-6), float(positive.max()) * 1.1

        upper = float(values.max()) * 1.1
        if normalized:
            return 0.0, max(upper, 1.1)
        return 0.0, max(upper, 1.05)

    @staticmethod
    def _apply_rich_club_axis_scaling(
        ax: plt.Axes,
        max_k: int,
        y_limits: tuple[float, float],
        xscale: str,
        yscale: str,
    ) -> None:
        if xscale == "log":
            ax.set_xscale("symlog", linthresh=1)
        elif xscale != "linear":
            raise ValueError("xscale must be either 'linear' or 'log'.")

        if yscale == "log":
            ax.set_yscale("log")
        elif yscale != "linear":
            raise ValueError("yscale must be either 'linear' or 'log'.")

        ax.set_xlim(0, max(1, max_k))
        ax.set_ylim(*y_limits)

    @staticmethod
    def _apply_static_axis_scaling(ax: plt.Axes, xscale: str, yscale: str) -> None:
        if xscale == "log":
            ax.set_xscale("symlog", linthresh=1)
        elif xscale != "linear":
            raise ValueError("xscale must be either 'linear' or 'log'.")

        if yscale == "log":
            ax.set_yscale("log")
        elif yscale != "linear":
            raise ValueError("yscale must be either 'linear' or 'log'.")

    def _save_animation(
        self,
        animation: FuncAnimation,
        fig: plt.Figure,
        filename: str,
        fps: int,
    ) -> Path:
        if fps <= 0:
            raise ValueError("fps must be strictly positive.")

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / filename
        animation.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return path

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Figure saved to %s", path)
        return path
