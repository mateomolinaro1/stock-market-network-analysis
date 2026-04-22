from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.analytics.analytics import Analytics
from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder
from dotenv import load_dotenv
import logging
import sys

# Config
load_dotenv()
config = Config()
logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Data
data_manager = DataManager(config=config)
data_manager.load_data()

# Analytics
an = Analytics(config=config, data=data_manager)
an.get_analytics()

# Correlation
estimator = RollingCorrelationEstimator(
    lookback=config.target_variable_rolling_window
)
corr_cache = estimator.compute_rolling(data_manager.asset_returns)
logger.info(f"Computed {len(corr_cache)} rolling correlation matrices")

sample_date = list(corr_cache.keys())[len(corr_cache) // 2]
sample_corr = corr_cache[sample_date]
logger.info(f"Sample correlation matrix at {sample_date}: shape={sample_corr.shape}")

# Graph
builder = ThresholdGraphBuilder(use_absolute_threshold=True, keep_sign=True)
sample_graph = builder.build(corr=sample_corr, threshold=0.75)
logger.info(
    f"Graph at {sample_date}: {sample_graph.number_of_nodes()} nodes, "
    f"{sample_graph.number_of_edges()} edges"
)