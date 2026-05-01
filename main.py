from dotenv import load_dotenv
import logging
import sys

from tqdm.auto import tqdm

from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.analytics.analytics import Analytics
from stock_mkt_network_analysis.experiments.run_wf_cv import build_feature_pipeline, run_cv
from stock_mkt_network_analysis.utils.corr_cache import compute_corr_cache_key
from stock_mkt_network_analysis.utils.ts_cache import compute_ts_cache_key
from stock_mkt_network_analysis.utils.graph_features_cache import compute_graph_features_cache_key
from stock_mkt_network_analysis.utils.node_features_cache import compute_node_features_cache_key
from stock_mkt_network_analysis.experiments.timing_backtest import run_oracle_timing_backtest
from stock_mkt_network_analysis.experiments.prediction_timing_backtest import run_prediction_timing_backtest
from stock_mkt_network_analysis.experiments.cross_sectional_backtest import run_network_cross_sectional_backtest


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = Config()
    data_manager = DataManager(config=config)
    feature_pipeline = build_feature_pipeline(config)

    # -------------------------------------------------------------
    # Main pipeline — 7 steps tracked by tqdm
    # -------------------------------------------------------------
    with tqdm(total=7, desc="Pipeline", unit="step", position=0) as pbar:

        # ── Step 1 — Data ─────────────────────────────────────
        pbar.set_description("Loading data")
        data_manager.load_data()
        # Analytics must be constructed after load_data so aligned_df is populated
        an = Analytics(config=config, data=data_manager)
        pbar.update(1)

        # ── Step 2 — Feature pipeline cache ───────────────────
        pbar.set_description("Precomputing feature pipeline cache")
        corr_cache_key          = compute_corr_cache_key(config, data_manager.network_returns)
        ts_cache_key            = compute_ts_cache_key(config, data_manager.network_returns)
        graph_features_cache_key = compute_graph_features_cache_key(config, data_manager.network_returns)
        node_features_cache_key  = compute_node_features_cache_key(config, data_manager.network_returns)
        feature_pipeline.precompute_cache(
            returns=data_manager.network_returns,
            market_returns=data_manager.mkt_returns,
            risk_free_returns=data_manager.rf_returns,
            load_or_compute_corr=config.load_or_compute_corr,
            save_corr=config.save_corr,
            corr_cache_key=corr_cache_key,
            s3_bucket=config.bucket_name,
            s3_region=config.region,
            load_or_compute_ts=config.load_or_compute_ts,
            save_ts=config.save_ts,
            ts_cache_key=ts_cache_key,
            load_or_compute_graph_features=config.load_or_compute_graph_features,
            save_graph_features=config.save_graph_features,
            graph_features_cache_key=graph_features_cache_key,
            threshold_grid=config.threshold_grid,
            load_or_compute_node_features=config.load_or_compute_node_features,
            save_node_features=config.save_node_features,
            node_features_cache_key=node_features_cache_key,
            s3=data_manager.aws.s3,
        )
        pbar.update(1)

        # ── Step 3 — Walk-forward CV ───────────────────────────
        pbar.set_description("Walk-forward CV")
        cv_result         = run_cv(data_manager, feature_pipeline, config)
        predictions       = cv_result.predictions
        selection_history = cv_result.selection_history
        oos_metrics       = cv_result.oos_metrics
        pbar.update(1)

        # ── Step 4 — Analytics ────────────────────────────────
        # pbar.set_description("Analytics")
        # an.get_analytics(corr_cache=feature_pipeline._corr_cache)
        # pbar.update(1)

        ######################################################
        ### BACKTEST #########################################
        ######################################################

        # ── Step 5 — Oracle timing backtest ───────────────────
        # Perfect-foresight upper bound: invest in market when
        # regime=0, switch to risk-free for the 21-day crisis
        # window whenever the oracle target fires.
        pbar.set_description("Backtest — oracle timing")
        oracle_output_dir = config.ROOT_DIR / "outputs" / "figures" / "backtest_oracle_timing"
        run_oracle_timing_backtest(config, data_manager, oracle_output_dir)
        pbar.update(1)

        # ── Step 6 — Prediction-based timing backtest ─────────
        # For each (model, feature_mode) pair from the walk-forward CV,
        # converts y_pred_binary into a market-timing signal identical in
        # structure to the oracle: predicted regime=1 → hold risk-free for
        # TARGET_VARIABLE_ROLLING_WINDOW days, regime=0 → hold market.
        # Saves per-model plots + a combined cumulative chart.
        pbar.set_description("Backtest — prediction timing")
        prediction_output_dir = config.ROOT_DIR / "outputs" / "figures" / "backtest_prediction_timing"
        run_prediction_timing_backtest(config, data_manager, predictions, prediction_output_dir)
        pbar.update(1)

        # ── Step 7 — Cross-sectional network feature backtest ──
        # Use asset-level (node) network features as cross-sectional
        # signals for a long-only stock-selection strategy.
        # For each (feature, threshold) pair, runs long-top and long-bottom
        # equal-weight portfolios with a 1-day implementation lag.
        # Saves per-feature plots in output_dir/[feature_name]/threshold_[t]/
        # and a combined cumulative chart in output_dir/combined/threshold_[t]/
        pbar.set_description("Backtest — cross-sectional network")
        cs_output_dir = config.ROOT_DIR / "outputs" / "figures" / "backtest_cross_sectional"
        run_network_cross_sectional_backtest(
            config, data_manager,
            feature_pipeline._node_feature_cache,
            config.threshold_grid,
            cs_output_dir,
        )
        pbar.update(1)


if __name__ == "__main__":
    main()