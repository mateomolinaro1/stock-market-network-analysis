from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import pandas as pd
from joblib import Parallel, delayed
import logging
import sys

from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.cv.feature_pipeline import RollingNetworkFeaturePipeline
from stock_mkt_network_analysis.cv.wf_cv import SimpleRollingWalkForwardCV
from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.network.correlation import RollingCorrelationEstimator
from stock_mkt_network_analysis.network.feature_extractor import BasicNetworkFeatureExtractor
from stock_mkt_network_analysis.network.graph_builder import ThresholdGraphBuilder
from stock_mkt_network_analysis.utils.ml_metrics import get_scoring_func
from stock_mkt_network_analysis.time_series.adaptive_time_series_feature_extractor import (
    AdaptiveTimeSeriesFeatureExtractor,
)


@dataclass
class CVRunResult:
    predictions: pd.DataFrame
    selection_history: pd.DataFrame
    oos_metrics: pd.DataFrame


def _run_single_mode(
    mode, feature_pipeline, model_grid, threshold_grid,
    scoring_func, outer_train_size, val_size, target_horizon,
    expanding_or_rolling, returns, target, outer_test_dates,
    aws_s3, cv_config,
):
    _logger = logging.getLogger(__name__)
    _logger.info(f"Starting CV | feature_mode={mode}")
    runner = SimpleRollingWalkForwardCV(
        feature_pipeline=feature_pipeline,
        model_grid=model_grid,
        threshold_grid=threshold_grid,
        scoring_func=scoring_func,
        outer_train_size=outer_train_size,
        val_size=val_size,
        target_horizon=target_horizon,
        feature_mode=mode,
        expanding_or_rolling=expanding_or_rolling,
        evict_cache=False,  # disabled: shared pipeline cache must not be modified concurrently
    )
    return runner.run(
        returns=returns,
        target=target,
        outer_test_dates=outer_test_dates,
        aws_s3=aws_s3 if mode == "all" else None,
        cv_config=cv_config if mode == "all" else None,
    )


def build_feature_pipeline(config: Config) -> RollingNetworkFeaturePipeline:
    """Construct the feature pipeline objects. Caller is responsible for precompute_cache()."""
    correlation_estimator = RollingCorrelationEstimator(
        lookback=config.lookback_corr, halflife=config.halflife_corr
    )
    graph_builder = ThresholdGraphBuilder(use_absolute_threshold=True, keep_sign=True)
    feature_extractor = BasicNetworkFeatureExtractor()
    time_series_feature_extractor = AdaptiveTimeSeriesFeatureExtractor.from_config(config)

    pipeline = RollingNetworkFeaturePipeline(
        correlation_estimator=correlation_estimator,
        graph_builder=graph_builder,
        feature_extractor=feature_extractor,
        time_series_feature_extractor=time_series_feature_extractor,
    )
    return pipeline


def run_cv(data_manager: DataManager, feature_pipeline: RollingNetworkFeaturePipeline, config: Config) -> CVRunResult:
    """
    Run the full walk-forward CV given a pre-built, pre-cached feature pipeline.

    Separating pipeline construction from CV logic allows main.py to share
    the precomputed correlation cache with the analytics step.
    """
    logger = logging.getLogger(__name__)

    outer_train_size = config.inner_train_size
    val_size = config.inner_val_size
    threshold_grid = config.threshold_grid
    model_grid = config.model_grid

    returns = data_manager.network_returns
    dates = returns.index

    # IMPORTANT:
    # target must already be forward-defined, i.e.
    # target[t] = future event starting at t
    target = data_manager.target_variable_to_predict.squeeze().dropna()

    # ------------------------------------------------------------------
    # Node-level network features (date x stock_id x metric)
    # ------------------------------------------------------------------
    logger.info("Computing node-level network features...")
    outputs_dir = config.ROOT_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    for threshold in threshold_grid:
        node_features = feature_pipeline.compute_node_features(returns=returns, threshold=threshold)
        out_path = outputs_dir / f"node_features_threshold_{threshold}.parquet"
        node_features.to_parquet(out_path)
        logger.info(f"Node features saved to {out_path} — shape: {node_features.shape}")

    # ------------------------------------------------------------------
    # Diagnostic: feature–target correlations
    # ------------------------------------------------------------------
    X_all = feature_pipeline.make_features(returns=returns, threshold=threshold_grid[0])
    y_all = target.reindex(X_all.index).dropna()
    X_aligned = X_all.reindex(y_all.index).dropna(how="all", axis=1)
    X_aligned = X_aligned.loc[:, X_aligned.std() > 0]
    corr = X_aligned.corrwith(y_all).dropna().sort_values()
    print("\n=== Feature-target Pearson correlations ===")
    print("Bottom 10 (most anti-predictive):")
    print(corr.head(10).to_string())
    print("\nTop 10 (most predictive):")
    print(corr.tail(10).to_string())
    print(f"\nTotal features: {len(corr)} | Positive: {(corr > 0).sum()} | Negative: {(corr < 0).sum()}")

    # ------------------------------------------------------------------
    # Walk-forward CV — one run per feature mode, parallelised by threads
    # ------------------------------------------------------------------
    start_oos = config.lookback_corr + outer_train_size + val_size + 1
    outer_test_dates = dates[start_oos:]

    n_jobs = len(config.feature_modes)
    logger.info(f"Running CV for {n_jobs} feature mode(s) in parallel: {config.feature_modes}")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_single_mode)(
            mode, feature_pipeline, model_grid, threshold_grid,
            get_scoring_func(config.scoring_metric),
            outer_train_size, val_size, config.target_variable_rolling_window,
            config.expanding_or_rolling, returns, target, outer_test_dates,
            data_manager.aws.s3, config,
        )
        for mode in config.feature_modes
    )

    all_predictions = []
    all_selection = []
    all_oos_metrics = []
    for mode, result in zip(config.feature_modes, results):
        all_predictions.append(result.predictions.reset_index())
        all_selection.append(result.selection_history.reset_index())
        all_oos_metrics.append(result.oos_metrics.assign(feature_mode=mode))

    predictions = pd.concat(all_predictions, ignore_index=True)
    selection_history = pd.concat(all_selection, ignore_index=True)
    oos_metrics = pd.concat(all_oos_metrics, ignore_index=True).set_index(["feature_mode", "model"])

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    vs = predictions["validation_score"]
    n_total = len(vs)
    n_valid = vs.notna().sum()
    n_dummy = (predictions["best_threshold"].isna() & vs.isna()).sum()
    n_skipped = vs.isna().sum() - n_dummy
    print(f"\nValidation score coverage: {n_valid}/{n_total} non-NaN ({100*n_valid/n_total:.1f}%)")
    print(f"  NaN breakdown — DummyClassifier fallback: {n_dummy} | skipped dates: {n_skipped}")

    print("\n=== Mean validation score (in-sample model selection) ===")
    val_summary = (
        predictions.dropna(subset=["validation_score"])
        .groupby(["feature_mode", "model"])["validation_score"]
        .agg(mean="mean", std="std", n="count")
    )
    print(val_summary.to_string())

    print("\n=== OOS metrics (out-of-sample) ===")
    print(oos_metrics.to_string())

    print("\nSelection history head:")
    print(selection_history.head())
    print()

    return CVRunResult(
        predictions=predictions,
        selection_history=selection_history,
        oos_metrics=oos_metrics,
    )


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    config = Config()

    logger.info("Loading data...")
    data_manager = DataManager(config=config)
    data_manager.load_data()
    logger.info("Data loaded successfully")

    logger.info("Precomputing feature pipeline cache...")
    feature_pipeline = build_feature_pipeline(config)
    feature_pipeline.precompute_cache(
        returns=data_manager.network_returns,
        market_returns=data_manager.mkt_returns,
        risk_free_returns=data_manager.rf_returns,
    )
    logger.info("Feature pipeline cache precomputed successfully")

    result = run_cv(data_manager, feature_pipeline, config)
    # result.predictions, result.selection_history, result.oos_metrics are available here


if __name__ == "__main__":
    main()