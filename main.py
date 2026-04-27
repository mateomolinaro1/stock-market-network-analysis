from dotenv import load_dotenv
import logging
import sys

from tqdm.auto import tqdm

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

config = Config()
data_manager = DataManager(config=config)
an = Analytics(config=config, data=data_manager)
feature_pipeline = build_feature_pipeline(config)

# -------------------------------------------------------------
# Main pipeline — 4 steps tracked by tqdm
# -------------------------------------------------------------
with tqdm(total=4, desc="Pipeline", unit="step", position=0) as pbar:

    # Step 1 — Data
    pbar.set_description("Loading data")
    data_manager.load_data()
    pbar.update(1)

    # Step 2 — Feature pipeline cache (rolling correlations + TS features)
    pbar.set_description("Precomputing feature pipeline cache")
    feature_pipeline.precompute_cache(
        returns=data_manager.network_returns,
        market_returns=data_manager.mkt_returns,
        risk_free_returns=data_manager.rf_returns,
    )
    pbar.update(1)

    ######################################################
    ### CV part ##########################################
    ######################################################
    pbar.set_description("Walk-forward CV")
    cv_result = run_cv(data_manager, feature_pipeline, config)
    predictions       = cv_result.predictions
    selection_history = cv_result.selection_history
    oos_metrics       = cv_result.oos_metrics
    pbar.update(1)

    ######################################################
    ### Analytics ########################################
    ######################################################
    pbar.set_description("Analytics")
    an.get_analytics(corr_cache=feature_pipeline._corr_cache)
    pbar.update(1)