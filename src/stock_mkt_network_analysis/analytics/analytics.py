"""
A module for performing various analytics on stock market data.
"""
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.utils.market_metric_utils import Metrics
from stock_mkt_network_analysis.analytics.visualization import Vizu
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