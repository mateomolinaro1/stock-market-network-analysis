from __future__ import annotations

from dotenv import load_dotenv

from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.cv.feature_pipeline import RollingNetworkFeaturePipeline
from stock_mkt_network_analysis.cv.wf_cv import SimpleRollingWalkForwardCV
from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.feature_extractor import BasicNetworkFeatureExtractor
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder
from stock_mkt_network_analysis.time_series.adaptive_time_series_feature_extractor import (
    AdaptiveTimeSeriesFeatureExtractor,
)
from stock_mkt_network_analysis.utils.ml_metrics import get_scoring_func
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
OUTER_TRAIN_SIZE = config.inner_train_size
VAL_SIZE = config.inner_val_size
LOOKBACK = config.lookback_corr
THRESHOLD_GRID = config.threshold_grid
MODEL_GRID = config.model_grid
# # Each entry: (base_estimator, param_grid) — each model independently tunes its own params
# MODEL_GRID = [
#     (
#         LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000),
#         [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
#     ),
#     (
#         RandomForestClassifier(class_weight="balanced", random_state=42),
#         [{"n_estimators": 100, "max_depth": 3}, {"n_estimators": 100, "max_depth": 5}],
#     ),
#     (
#         GradientBoostingClassifier(random_state=42),
#         [{"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05},
#          {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}],
#     ),
# ]

def main():
    # -----------------------------
    # Data
    # -----------------------------
    logger.info("Loading data...")
    data_manager = DataManager(config=config)
    data_manager.load_data()
    logger.info("Data loaded successfully")

    returns = data_manager.network_returns
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
    time_series_feature_extractor = AdaptiveTimeSeriesFeatureExtractor.from_config(config)

    feature_pipeline = RollingNetworkFeaturePipeline(
        correlation_estimator=correlation_estimator,
        graph_builder=graph_builder,
        feature_extractor=feature_extractor,
        time_series_feature_extractor=time_series_feature_extractor,
    )
    logger.info("Precomputing feature pipeline cache...")
    feature_pipeline.precompute_cache(
        returns=returns,
        market_returns=data_manager.mkt_returns,
        risk_free_returns=data_manager.rf_returns,
    )
    logger.info("Feature pipeline cache precomputed successfully")

    start_oos = LOOKBACK + OUTER_TRAIN_SIZE + VAL_SIZE + 1
    outer_test_dates = dates[start_oos:]

    runner = SimpleRollingWalkForwardCV(
        feature_pipeline=feature_pipeline,
        model_grid=MODEL_GRID,
        threshold_grid=THRESHOLD_GRID,
        scoring_func=get_scoring_func(config.scoring_metric),
        outer_train_size=OUTER_TRAIN_SIZE,
        val_size=VAL_SIZE,
        target_horizon=config.target_variable_rolling_window,
    )

    result = runner.run(
        returns=returns,
        target=target,
        outer_test_dates=outer_test_dates,
        aws_s3=data_manager.aws.s3,
        cv_config=config,
    )

    print("Predictions head:")
    print(result.predictions.head())
    print()
    print(result.predictions["validation_score"].describe())

    print("Selection history head:")
    print(result.selection_history.head())
    print()

    # if len(result.predictions) > 0:
    #     print("\nOOS ROC AUC by model:")
    #     for model_name, grp in result.predictions.groupby("model"):
    #         if grp["y_true"].nunique() > 1:
    #             auc = roc_auc_score(grp["y_true"], grp["y_pred"])
    #             print(f"  {model_name}: {auc:.4f}  (n={len(grp)})")


if __name__ == "__main__":
    main()