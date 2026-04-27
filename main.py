from dotenv import load_dotenv
import logging
import sys

from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.analytics.analytics import Analytics
from stock_mkt_network_analysis.experiments.run_wf_cv import build_feature_pipeline, run_cv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Config & data
# -------------------------------------------------------------
config = Config()
data_manager = DataManager(config=config)
data_manager.load_data()

# -------------------------------------------------------------
# Shared feature pipeline  (correlation cache built once here,
# reused by both CV and Analytics — no redundant recomputation)
# -------------------------------------------------------------
logger.info("Precomputing feature pipeline cache...")
feature_pipeline = build_feature_pipeline(config)
feature_pipeline.precompute_cache(
    returns=data_manager.network_returns,
    market_returns=data_manager.mkt_returns,
    risk_free_returns=data_manager.rf_returns,
)
logger.info("Feature pipeline cache precomputed successfully")

######################################################
### CV part ##########################################
######################################################
run_cv(data_manager, feature_pipeline, config)

######################################################
### Analytics ########################################
######################################################
an = Analytics(config=config, data=data_manager)
an.get_analytics(corr_cache=feature_pipeline._corr_cache)