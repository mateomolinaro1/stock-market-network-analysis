"""
A module for performing various analytics on stock market data.
"""
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

    def get_analytics(self)->None:
        """
        Complete the analytics pipeline by computing necessary metrics and preparing data for visualization.
        :return:
        """
        self._get_data_for_analytics()
        self._get_plot_target_variable()
        self._get_plot_target_variable_with_cum_ret()
        self._get_plot_raw_target_variable()

    def get_network_animations(
            self,
            start_date: str | None = None,
            end_date: str | None = None,
            threshold: float | None = None,
            degree_threshold: int | None = None,
            max_frames: int | None = None,
            normalize_degree_counts: bool = False,
            degree_xscale: str = "linear",
            degree_yscale: str = "linear",
            degree_y_max_quantile: float | None = None,
            degree_plot_kind: str = "hist",
            normalize_rich_club: bool = False,
            n_random_reference: int = 10,
            random_swaps_per_edge: int = 5,
            rich_club_xscale: str = "linear",
            rich_club_yscale: str = "linear",
    ) -> dict[str, object]:
        """
        Generate animated diagnostics for rolling correlation networks.

        If no period is provided, animations are generated on all eligible dates.
        """
        if threshold is None and not self.config.threshold_grid:
            raise ValueError("A threshold must be provided when config.threshold_grid is empty.")

        threshold = threshold if threshold is not None else self.config.threshold_grid[0]

        animator = RollingNetworkAnimator(
            correlation_estimator=RollingCorrelationEstimator(lookback=self.config.lookback_corr),
            graph_builder=ThresholdGraphBuilder(use_absolute_threshold=True, keep_sign=True),
            threshold=threshold,
            figures_dir=self.config.ROOT_DIR / "outputs" / "figures",
        )

        returns = self.data.network_returns
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
        )

        return {
            "degree_distribution": degree_path,
            "rich_club": rich_club_path,
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
