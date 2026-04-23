from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv

from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.cv.feature_pipeline import RollingNetworkFeaturePipeline
from stock_mkt_network_analysis.cv.wf_cv import SimpleRollingWalkForwardCV
from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.feature_extractor import BasicNetworkFeatureExtractor
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder
from stock_mkt_network_analysis.utils.ml_metrics import safe_roc_auc
import logging
import sys

load_dotenv()
config = Config()
logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# -----------------------------
# Simple walk-forward parameters
# -----------------------------
OUTER_TRAIN_SIZE = config.inner_train_size   # e.g. 1 year of daily data
VAL_SIZE = config.inner_val_size            # e.g. 1 quarter of daily data
LOOKBACK = config.lookback_corr              # lookback for rolling correlation estimation (e.g. 1 year)
THRESHOLD_GRID = config.threshold_grid  # example thresholds for graph construction
LOGIT_PARAM_GRID = config.logit_param_grid


def main():
    # -----------------------------
    # Data
    # -----------------------------
    logger.info("Loading data...")
    data_manager = DataManager(config=config)
    data_manager.load_data()
    logger.info("Data loaded successfully")

    asset_cols = data_manager.asset_returns.columns
    returns = data_manager.aligned_df[asset_cols]
    dates = returns.index

    # IMPORTANT:
    # target must already be forward-defined, i.e.
    # target[t] = future event starting at t
    target = data_manager.target_variable_to_predict.squeeze().dropna()

    # -----------------------------
    # Feature pipeline
    # -----------------------------
    correlation_estimator = RollingCorrelationEstimator(lookback=LOOKBACK)
    graph_builder = ThresholdGraphBuilder(use_absolute_threshold=True, keep_sign=True)
    feature_extractor = BasicNetworkFeatureExtractor()

    feature_pipeline = RollingNetworkFeaturePipeline(
        correlation_estimator=correlation_estimator,
        graph_builder=graph_builder,
        feature_extractor=feature_extractor,
    )
    logger.info("Precomputing feature pipeline cache...")
    feature_pipeline.precompute_cache(returns)
    logger.info("Feature pipeline cache precomputed successfully")

    # -----------------------------
    # Model
    # -----------------------------
    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
    )

    # -----------------------------
    # OOS start
    # Need enough data for:
    # lookback + outer_train_size + val_size
    # -----------------------------
    start_oos = LOOKBACK + OUTER_TRAIN_SIZE + VAL_SIZE + 1
    outer_test_dates = dates[start_oos:]

    # -----------------------------
    # Simple rolling walk-forward
    # -----------------------------
    runner = SimpleRollingWalkForwardCV(
        feature_pipeline=feature_pipeline,
        model=model,
        threshold_grid=THRESHOLD_GRID,
        hyperparameter_grid=LOGIT_PARAM_GRID,
        scoring_func=safe_roc_auc,
        outer_train_size=OUTER_TRAIN_SIZE,
        val_size=VAL_SIZE,
    )

    result = runner.run(
        returns=returns,
        target=target,
        outer_test_dates=outer_test_dates,
    )

    print("Predictions head:")
    print(result.predictions.head())
    print()

    print("Selection history head:")
    print(result.selection_history.head())
    print()

    if len(result.predictions) > 0 and result.predictions["y_true"].nunique() > 1:
        auc = roc_auc_score(
            result.predictions["y_true"],
            result.predictions["y_pred"],
        )
        print(f"OOS ROC AUC: {auc:.4f}")


if __name__ == "__main__":
    main()