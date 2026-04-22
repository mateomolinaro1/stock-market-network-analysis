from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.cv.feature_pipeline import RollingNetworkFeaturePipeline
from stock_mkt_network_analysis.cv.nested_wf_cv import NestedWalkForwardCV
from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.feature_extractor import BasicNetworkFeatureExtractor
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder
from stock_mkt_network_analysis.utils.ml_metrics import safe_roc_auc
from dotenv import load_dotenv
load_dotenv()
config = Config()

def main():
    # Data
    data_manager = DataManager(config=config)
    data_manager.load_data()

    asset_cols = data_manager.asset_returns.columns
    returns = data_manager.aligned_df[asset_cols]
    dates = returns.index

    target = data_manager.target_variable_to_predict.squeeze().dropna()

    correlation_estimator = RollingCorrelationEstimator(lookback=config.lookback_target_and_corr)
    graph_builder = ThresholdGraphBuilder(use_absolute_threshold=True, keep_sign=True)
    feature_extractor = BasicNetworkFeatureExtractor()

    feature_pipeline = RollingNetworkFeaturePipeline(
        correlation_estimator=correlation_estimator,
        graph_builder=graph_builder,
        feature_extractor=feature_extractor,
    )

    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
    )
    start_oos = config.lookback_target_and_corr + config.inner_train_size + config.inner_val_size + 1
    outer_test_dates = dates[start_oos:]

    runner = NestedWalkForwardCV(
        feature_pipeline=feature_pipeline,
        model=model,
        threshold_grid=config.threshold_grid,
        hyperparameter_grid=config.logit_param_grid,
        scoring_func=safe_roc_auc,
        inner_train_size=config.inner_train_size,
        inner_val_size=config.inner_val_size,
        inner_step_size=config.inner_step_size,
    )

    result = runner.run(
        returns=returns,
        target=target,
        outer_test_dates=outer_test_dates,
    )

    print(result.predictions.head())
    print(result.selection_history.head())

    if len(result.predictions) > 0 and result.predictions["y_true"].nunique() > 1:
        auc = roc_auc_score(
            result.predictions["y_true"],
            result.predictions["y_pred"],
        )
        print(f"OOS ROC AUC: {auc:.4f}")


if __name__ == "__main__":
    main()