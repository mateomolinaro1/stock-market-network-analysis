import pandas as pd
from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.analytics.visualization import Vizu
from dotenv import load_dotenv
load_dotenv()
config = Config()

data_manager = DataManager(config=config)
data_manager.load_data()
# Build plots target variable (rolling cum perf mkt, rolling max dd, target variable)
Vizu.plot_time_series(
    df=data_manager.aligned_df,
    x_index=True,
    x_col=None,
    y_col=data_manager.target_variable.columns.to_list(),
    y2_col=data_manager.rolling_raw_target_variable.columns.to_list(),
    title="Target variable over time",
    xlabel="Date",
    ylabel="Target variable",
    y2_label="Rolling raw target variable",
    saving_path=config.ROOT_DIR/"outputs"/"figures"/"target_variable_over_time.png"
)