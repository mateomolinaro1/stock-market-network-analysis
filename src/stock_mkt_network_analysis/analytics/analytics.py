"""
A module for performing various analytics on stock market data.
"""
from typing import Dict, Optional

import pandas as pd

from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.utils.market_metric_utils import Metrics
from stock_mkt_network_analysis.analytics.visualization import Vizu
from stock_mkt_network_analysis.analytics.network_animations import RollingNetworkAnimator
from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder
from stock_mkt_network_analysis.utils.config import Config
import logging

logger = logging.getLogger(__name__)

class Analytics:
    def __init__(self, config: Config, data: DataManager):
        self.data = data
        self.config = config

        # Data for analytics
        self.aligned_df = self.data.aligned_df
        self.rolling_raw_target_variable = self.data.rolling_raw_target_variable
        self.mkt_cumulative_returns = None

    def get_analytics(
        self,
        corr_cache: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
    ) -> None:
        """
        Complete the analytics pipeline by computing necessary metrics and preparing data for visualization.

        Pass corr_cache (e.g. feature_pipeline._corr_cache) to skip rolling-correlation
        recomputation inside the network animations — the dominant cost.
        """
        self._get_data_for_analytics()
        self._get_plot_target_variable()
        self._get_plot_target_variable_with_cum_ret()
        self._get_plot_raw_target_variable()
        self._get_network_animations(corr_cache=corr_cache)

    def _get_network_animations(
            self,
            start_date: str | None = None,
            end_date: str | None = None,
            threshold: float | None = None,
            degree_threshold: int | None = None,
            max_frames: int | None = None,
            normalize_degree_counts: bool | None = None,
            degree_xscale: str | None = None,
            degree_yscale: str | None = None,
            degree_y_max_quantile: float | None = None,
            degree_plot_kind: str | None = None,
            degree_log_bins: int | None = None,
            normalize_rich_club: bool | None = None,
            n_random_reference: int | None = None,
            random_swaps_per_edge: int | None = None,
            rich_club_xscale: str | None = None,
            rich_club_yscale: str | None = None,
            regime_max_dates: int | None = None,
            show_power_law: bool | None = None,
            power_law_min_degree: float | None = None,
            power_law_max_degree: float | None = None,
            degree_mixing_n_bins: int | None = None,
            degree_mixing_max_points_per_regime: int | None = None,
            corr_cache: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
    ) -> dict[str, object]:
        """
        Generate animated diagnostics for rolling correlation networks.

        If no period is provided, animations are generated on all eligible dates.
        Parameters left to None use config.network_analytics, which is loaded
        from config/run_pipeline_config.json.
        corr_cache: pre-computed {date: corr_matrix} dict; when provided, rolling
        correlations are not recomputed (saves the dominant runtime cost).
        """
        if threshold is None and not self.config.threshold_grid:
            raise ValueError("A threshold must be provided when config.threshold_grid is empty.")

        plot_config = self.config.network_analytics
        threshold = threshold if threshold is not None else self.config.threshold_grid[0]
        degree_threshold = degree_threshold if degree_threshold is not None else plot_config.degree_threshold
        max_frames = max_frames if max_frames is not None else plot_config.max_frames
        normalize_degree_counts = (
            normalize_degree_counts
            if normalize_degree_counts is not None
            else plot_config.normalize_degree_counts
        )
        degree_xscale = degree_xscale if degree_xscale is not None else plot_config.degree_xscale
        degree_yscale = degree_yscale if degree_yscale is not None else plot_config.degree_yscale
        degree_y_max_quantile = (
            degree_y_max_quantile
            if degree_y_max_quantile is not None
            else plot_config.degree_y_max_quantile
        )
        degree_plot_kind = degree_plot_kind if degree_plot_kind is not None else plot_config.degree_plot_kind
        degree_log_bins = degree_log_bins if degree_log_bins is not None else plot_config.degree_log_bins
        normalize_rich_club = (
            normalize_rich_club
            if normalize_rich_club is not None
            else plot_config.normalize_rich_club
        )
        n_random_reference = (
            n_random_reference
            if n_random_reference is not None
            else plot_config.n_random_reference
        )
        random_swaps_per_edge = (
            random_swaps_per_edge
            if random_swaps_per_edge is not None
            else plot_config.random_swaps_per_edge
        )
        rich_club_xscale = rich_club_xscale if rich_club_xscale is not None else plot_config.rich_club_xscale
        rich_club_yscale = rich_club_yscale if rich_club_yscale is not None else plot_config.rich_club_yscale
        regime_max_dates = regime_max_dates if regime_max_dates is not None else plot_config.regime_max_dates
        show_power_law = show_power_law if show_power_law is not None else plot_config.show_power_law
        power_law_min_degree = (
            power_law_min_degree
            if power_law_min_degree is not None
            else plot_config.power_law_min_degree
        )
        power_law_max_degree = (
            power_law_max_degree
            if power_law_max_degree is not None
            else plot_config.power_law_max_degree
        )
        degree_mixing_n_bins = (
            degree_mixing_n_bins
            if degree_mixing_n_bins is not None
            else plot_config.degree_mixing_n_bins
        )
        degree_mixing_max_points_per_regime = (
            degree_mixing_max_points_per_regime
            if degree_mixing_max_points_per_regime is not None
            else plot_config.degree_mixing_max_points_per_regime
        )

        animator = RollingNetworkAnimator(
            correlation_estimator=RollingCorrelationEstimator(
                lookback=self.config.lookback_corr, halflife=self.config.halflife_corr
            ),
            graph_builder=ThresholdGraphBuilder(use_absolute_threshold=True, keep_sign=True),
            threshold=threshold,
            figures_dir=self.config.ROOT_DIR / "outputs" / "figures",
            corr_cache=corr_cache,
        )

        returns = self.data.network_returns
        target = self.data.target_variable_to_predict.squeeze().dropna()
        degree_path = animator.animate_degree_distribution(
            returns=returns,
            start_date=start_date,
            end_date=end_date,
            max_frames=max_frames,
            normalize_counts=normalize_degree_counts,
            xscale=degree_xscale,
            yscale=degree_yscale,
            y_max_quantile=degree_y_max_quantile,
            plot_kind=degree_plot_kind,
            log_bins=degree_log_bins,
            target=target,
            show_power_law=show_power_law,
            power_law_min_degree=power_law_min_degree,
            power_law_max_degree=power_law_max_degree,
        )
        rich_club_path = animator.animate_rich_club(
            returns=returns,
            start_date=start_date,
            end_date=end_date,
            degree_threshold=degree_threshold,
            max_frames=max_frames,
            normalized=normalize_rich_club,
            n_random_reference=n_random_reference,
            random_swaps_per_edge=random_swaps_per_edge,
            xscale=rich_club_xscale,
            yscale=rich_club_yscale,
            target=target,
        )
        regime_degree_path = animator.plot_degree_distribution_by_regime(
            returns=returns,
            target=target,
            start_date=start_date,
            end_date=end_date,
            max_dates=regime_max_dates,
            xscale=degree_xscale,
            yscale=degree_yscale,
            plot_kind=degree_plot_kind,
            log_bins=degree_log_bins,
            normalize_counts=True,
        )
        regime_rich_club_path = animator.plot_rich_club_by_regime(
            returns=returns,
            target=target,
            start_date=start_date,
            end_date=end_date,
            max_dates=regime_max_dates,
            normalized=normalize_rich_club,
            n_random_reference=n_random_reference,
            random_swaps_per_edge=random_swaps_per_edge,
            xscale=rich_club_xscale,
            yscale=rich_club_yscale,
        )
        degree_mixing_paths = animator.plot_degree_mixing_scatter_per_regime(
            returns=returns,
            target=target,
            start_date=start_date,
            end_date=end_date,
            max_dates=regime_max_dates,
            n_bins=degree_mixing_n_bins,
            max_points_per_regime=degree_mixing_max_points_per_regime,
            xscale=degree_xscale,
            yscale=degree_yscale,
        )

        return {
            "degree_distribution": degree_path,
            "rich_club": rich_club_path,
            "degree_distribution_by_regime": regime_degree_path,
            "rich_club_by_regime": regime_rich_club_path,
            "degree_mixing_normal": degree_mixing_paths.get("normal"),
            "degree_mixing_tendu": degree_mixing_paths.get("tendu"),
        }


    def _get_data_for_analytics(self)->None:
        """
        Compute analytics such as rolling cumulative performance and rolling max drawdown.
        """
        self.mkt_cumulative_returns = Metrics.compute_cumulative_return(
            df=self.data.aligned_df[self.data.mkt_returns.columns.to_list()]
        )
        # Build a combined df that includes the forward-looking label used by the model
        self.analytics_df = self.aligned_df.join(
            self.data.target_variable_to_predict.rename(
                columns={self.config.target_variable: "target_to_predict"}
            ),
            how="left"
        )

    def _get_plot_target_variable(self)->None:
        """
        Uses the plot function in vizu
        :return:
        """
        Vizu.plot_time_series(
            df=self.analytics_df,
            x_index=True,
            x_col=None,
            y_col="target_to_predict",
            y2_col=self.data.rolling_raw_target_variable.columns.to_list(),
            title="Forward-looking target variable over time",
            xlabel="Date",
            ylabel="Target to predict (0/1)",
            y2_label="Forward max drawdown (raw)",
            saving_path=self.config.ROOT_DIR / "outputs" / "figures" / "target_variable_over_time.png",
            date_freq=self.config.data_freq
        )

    def _get_plot_target_variable_with_cum_ret(self)->None:
        """
        Uses the plot function in vizu
        :return:
        """
        Vizu.plot_time_series(
            df=self.analytics_df,
            x_index=True,
            x_col=None,
            y_col="target_to_predict",
            y2_col=self.data.mkt_cumulative_returns.columns.to_list(),
            title="Forward-looking target variable and cumulative return of bench",
            xlabel="Date",
            ylabel="Target to predict (0/1)",
            y2_label="Cum ret bench",
            saving_path=self.config.ROOT_DIR / "outputs" / "figures" / "target_variable_over_time_with_cum_ret.png",
            date_freq=self.config.data_freq
        )

    def _get_plot_raw_target_variable(self)->None:
        """
        Plot the raw target variable over time.
        :return:
        """
        Vizu.plot_time_series(
            df=self.analytics_df,
            x_index=True,
            x_col=None,
            y_col=self.data.mkt_cumulative_returns.columns.to_list(),
            y2_col=self.data.rolling_raw_target_variable.columns.to_list(),
            title="Cumulative market return and forward max drawdown over time",
            xlabel="Date",
            ylabel="Cumulative market return",
            y2_label="Forward max drawdown (raw)",
            saving_path=self.config.ROOT_DIR / "outputs" / "figures" / "cum_return_over_time.png",
            date_freq=self.config.data_freq
        )
