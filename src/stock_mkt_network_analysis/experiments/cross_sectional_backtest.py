"""
Cross-sectional network feature backtest.

For each node-level feature (degree, strength, …) and each correlation threshold,
runs two long-only equal-weight portfolios:
  - Long-top   : EW basket of stocks in the top    PERCENTILES_PORTFOLIOS[1] percentile
  - Long-bottom: EW basket of stocks in the bottom PERCENTILES_PORTFOLIOS[0] percentile

Node features are lagged by 1 trading day (implementation lag) before portfolio
construction to avoid look-ahead bias.

Saves per-feature plots to:
    output_dir / [feature_name] / threshold_[t] / *.png
Combined cumulative chart (all features per threshold):
    output_dir / combined / threshold_[t] / combined_cumulative.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_mkt_network_analysis.backtester.analysis import PerformanceAnalyser
from stock_mkt_network_analysis.backtester.backtest import Backtest
from stock_mkt_network_analysis.backtester.portfolio import EqualWeightingScheme
from stock_mkt_network_analysis.backtester.strategies import CrossSectionalPercentiles
from stock_mkt_network_analysis.backtester.visualization import Visualizer
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.utils.config import Config


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_backtest(
    signal_values: pd.DataFrame,
    returns: pd.DataFrame,
    config: Config,
    strategy_name: str,
) -> Backtest:
    """Long-only EW backtest driven by cross-sectional signal values."""
    strat = CrossSectionalPercentiles(
        returns=returns,
        signal_values=signal_values,
        percentiles_winsorization=tuple(config.cs_backtest_percentiles_winsorization),
    )
    strat.compute_signals_values()
    signals = strat.compute_signals(
        percentiles_portfolios=tuple(config.cs_backtest_percentiles_portfolios)
    )

    ew = EqualWeightingScheme(
        returns=returns,
        signals=signals,
        rebal_periods=0,
        portfolio_type="long_only",
    )
    ew.compute_weights()
    weights = ew.weights

    # diff() on fillna(0) weights: initial build cost is captured automatically
    turnover = weights.fillna(0).diff().abs().sum(axis=1).to_frame("turnover")

    bt = Backtest(
        returns=returns,
        weights=weights,
        turnover=turnover,
        transaction_costs=config.cs_backtest_transaction_costs_bps,
        strategy_name=strategy_name,
    )
    bt.run_backtest()
    return bt


def _make_perf(
    net_returns: pd.DataFrame,
    bench: pd.DataFrame,
    config: Config,
    rolling_window: int,
) -> PerformanceAnalyser:
    perf = PerformanceAnalyser(
        portfolio_returns=net_returns,
        bench_returns=bench,
        freq=config.data_freq,
    )
    perf.compute_cumulative_performance()
    perf.compute_metrics()
    perf.compute_rolling_metrics(window=rolling_window)
    perf.compute_yearly_metrics()
    return perf


def _scalar(val) -> float:
    if val is None:
        return float("nan")
    return float(val.iloc[0]) if hasattr(val, "iloc") else float(val)


# ── Terminal output ─────────────────────────────────────────────────────────


def _print_feature_summary(
    feature_name: str,
    threshold: float,
    perf_top: PerformanceAnalyser,
    perf_bot: PerformanceAnalyser,
    n_days: int,
) -> None:
    m_t = perf_top.metrics
    m_b = perf_bot.metrics
    header = f"  ─── {feature_name.upper()} | threshold={threshold} | {n_days} OOS days"
    print(f"\n{header}")
    print(f"  {'':20s} {'TotRet':>8}  {'AnnRet':>8}  {'Vol':>8}  {'SR':>6}  {'MaxDD':>9}")
    print(f"  {'Long-top':<20} {m_t['total_return']:>8.2%}  {m_t['annualized_return']:>8.2%}  "
          f"{m_t['annualized_volatility']:>8.2%}  {m_t['annualized_sharpe_ratio']:>6.2f}  "
          f"{m_t['max_drawdown']:>9.2%}")
    print(f"  {'Long-bottom':<20} {m_b['total_return']:>8.2%}  {m_b['annualized_return']:>8.2%}  "
          f"{m_b['annualized_volatility']:>8.2%}  {m_b['annualized_sharpe_ratio']:>6.2f}  "
          f"{m_b['max_drawdown']:>9.2%}")
    print(f"  {'Buy & Hold':<20} {_scalar(m_t['total_return_bench']):>8.2%}  "
          f"{_scalar(m_t['annualized_return_bench']):>8.2%}  "
          f"{_scalar(m_t['annualized_volatility_bench']):>8.2%}  "
          f"{_scalar(m_t['annualized_sharpe_ratio_bench']):>6.2f}  "
          f"{_scalar(m_t['max_drawdown_bench']):>9.2%}")


# ── Plots ───────────────────────────────────────────────────────────────────


def _plot_feature_cumulative(
    perf_top: PerformanceAnalyser,
    perf_bot: PerformanceAnalyser,
    feature_name: str,
    threshold: float,
    output_path: Path,
) -> None:
    """Both directions + benchmark on one chart."""
    fig, ax = plt.subplots(figsize=(14, 6))

    top_cum   = perf_top.cumulative_performance_base_100
    bot_cum   = perf_bot.cumulative_performance_base_100
    bench_cum = perf_top.bench_cumulative_perf_base_100

    ax.plot(top_cum.index,   top_cum.iloc[:, 0],   label="Long top",    linewidth=1.5)
    ax.plot(bot_cum.index,   bot_cum.iloc[:, 0],   label="Long bottom", linewidth=1.5, linestyle="--")
    for col in bench_cum.columns:
        ax.plot(bench_cum.index, bench_cum[col],
                label=f"Buy & Hold — {col}", linestyle=":", linewidth=2.0, color="black")

    m_t = perf_top.metrics
    m_b = perf_bot.metrics
    ax.set_title(
        f"{feature_name.replace('_', ' ').title()} — Threshold {threshold} — Cross-Sectional\n"
        f"Long Top:    ret={m_t['annualized_return']:.2%}, vol={m_t['annualized_volatility']:.2%}, "
        f"SR={m_t['annualized_sharpe_ratio']:.2f}, maxDD={m_t['max_drawdown']:.2%}\n"
        f"Long Bottom: ret={m_b['annualized_return']:.2%}, vol={m_b['annualized_volatility']:.2%}, "
        f"SR={m_b['annualized_sharpe_ratio']:.2f}, maxDD={m_b['max_drawdown']:.2%}\n"
        f"Benchmark:   ret={_scalar(m_t['annualized_return_bench']):.2%}, "
        f"vol={_scalar(m_t['annualized_volatility_bench']):.2%}, "
        f"SR={_scalar(m_t['annualized_sharpe_ratio_bench']):.2f}, "
        f"maxDD={_scalar(m_t['max_drawdown_bench']):.2%}",
        fontsize=10,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance (Base 100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_combined_cumulative(
    all_cum_perfs: Dict[str, pd.Series],
    bench_cum: pd.DataFrame,
    threshold: float,
    feature_names: List[str],
    output_path: Path,
) -> None:
    """One combined chart: all features × both directions + benchmark."""
    fig, ax = plt.subplots(figsize=(16, 7))

    # One color per feature (first 10 from tab10); solid = top, dashed = bottom
    color_map = {feat: cm.tab10(i / 10) for i, feat in enumerate(feature_names)}

    for label, series in all_cum_perfs.items():
        feature   = label.split(" (")[0]
        is_bottom = "(bottom)" in label
        ax.plot(
            series.index, series.values,
            label=label,
            linewidth=1.2,
            linestyle="--" if is_bottom else "-",
            color=color_map.get(feature, "gray"),
        )

    for col in bench_cum.columns:
        ax.plot(bench_cum.index, bench_cum[col],
                label=f"Buy & Hold — {col}", linestyle=":", linewidth=2.5, color="black")

    ax.set_title(
        f"Cross-Sectional Network Features — All Features — Threshold {threshold}\n"
        f"Cumulative Performance (Base 100)  |  solid = long-top, dashed = long-bottom",
        fontsize=11,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance (Base 100)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Summary table ──────────────────────────────────────────────────────────


def _print_summary_table(rows: list, threshold: float) -> None:
    """Print all strategies + benchmark sorted by Sharpe ratio descending."""
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda r: r["SR"], reverse=True)
    W = 76
    print()
    print("=" * W)
    print(f"  SUMMARY — threshold = {threshold}  (sorted by Sharpe Ratio ↓)")
    print("=" * W)
    print(f"  {'Strategy':<36} {'TotRet':>8}  {'AnnRet':>8}  {'Vol':>8}  {'SR':>6}  {'MaxDD':>9}")
    print("-" * W)
    for r in sorted_rows:
        tag = " ★" if r.get("is_bench") else ""
        print(
            f"  {r['Strategy'] + tag:<36} {r['TotRet']:>8.2%}  {r['AnnRet']:>8.2%}  "
            f"{r['Vol']:>8.2%}  {r['SR']:>6.2f}  {r['MaxDD']:>9.2%}"
        )
    print("=" * W)
    print("  ★ = benchmark")


# ── Main entry point ────────────────────────────────────────────────────────


def run_network_cross_sectional_backtest(
    config: Config,
    data_manager: DataManager,
    node_feature_cache: Dict[float, pd.DataFrame],
    threshold_grid: List[float],
    output_dir: Path,
) -> None:
    """
    Cross-sectional network feature backtest.

    For each (feature, threshold) pair runs long-top and long-bottom long-only
    EW portfolios based on the node-level network feature.  Node features are
    lagged by 1 trading day to avoid look-ahead bias.

    Expects data_manager.load_data() to have been called beforehand.
    Saves performance plots to output_dir.
    """
    logger = logging.getLogger(__name__)

    if not node_feature_cache:
        logger.warning("node_feature_cache is empty; skipping cross-sectional backtest.")
        return

    # ── OOS start — same burn-in as CV / oracle timing ─────────
    dates = data_manager.network_returns.index
    start_oos_loc = (
        config.lookback_corr
        + config.inner_train_size
        + config.inner_val_size
        + 1
    )
    if start_oos_loc >= len(dates):
        raise ValueError(
            f"start_oos_loc={start_oos_loc} exceeds available dates ({len(dates)}). "
            "Check LOOKBACK_CORR / INNER_TRAIN_SIZE / INNER_VAL_SIZE in config."
        )
    start_oos = dates[start_oos_loc]
    logger.info(
        "Cross-sectional backtest — OOS window: %s → %s  (%d trading days)",
        start_oos.date(), dates[-1].date(), len(dates) - start_oos_loc,
    )

    mkt_col     = config.mkt_benchmark_column
    returns_oos = data_manager.network_returns.loc[start_oos:]
    bench_oos   = data_manager.mkt_returns[[mkt_col]].reindex(returns_oos.index).ffill()
    rolling_win = config.cs_backtest_rolling_window_metrics

    for threshold in threshold_grid:
        if threshold not in node_feature_cache:
            logger.warning("Threshold %s not in node_feature_cache; skipping.", threshold)
            continue

        node_df = node_feature_cache[threshold]          # MultiIndex(date, stock_id)
        thr_str = str(threshold).replace(".", "_")

        all_cum_perfs: Dict[str, pd.Series] = {}
        bench_cum_ref: pd.DataFrame | None = None
        summary_rows:  list = []
        bench_row:     dict | None = None

        print()
        print("=" * 70)
        print(f"  CROSS-SECTIONAL BACKTEST — threshold = {threshold}")
        print("=" * 70)

        for feature_name in config.cs_backtest_node_features:
            if feature_name not in node_df.columns:
                logger.warning(
                    "Feature '%s' not in node cache for threshold %s; skipping.",
                    feature_name, threshold,
                )
                continue

            # ── Pivot → wide (date × stock_id), apply 1-day lag ─
            feature_wide    = node_df[feature_name].unstack("stock_id")
            feature_shifted = feature_wide.shift(1)      # implementation lag

            # Align to OOS date index and stock columns
            feature_oos = (
                feature_shifted
                .reindex(index=returns_oos.index)
                .reindex(columns=returns_oos.columns)
            )

            if feature_oos.notna().sum().sum() == 0:
                logger.warning(
                    "All NaN for feature '%s', threshold %s — skipping.",
                    feature_name, threshold,
                )
                continue

            # ── Long-top  (high feature value → buy) ────────────
            bt_top = _build_backtest(
                signal_values=feature_oos,
                returns=returns_oos,
                config=config,
                strategy_name=f"{feature_name}_top",
            )
            # ── Long-bottom (low feature value → buy, via negation)
            bt_bot = _build_backtest(
                signal_values=-feature_oos,
                returns=returns_oos,
                config=config,
                strategy_name=f"{feature_name}_bottom",
            )

            # ── Align both strategies to the same date range ────
            start_strat = max(bt_top.start_date, bt_bot.start_date)
            end_strat   = min(
                bt_top.cropped_portfolio_net_returns.index[-1],
                bt_bot.cropped_portfolio_net_returns.index[-1],
            )
            bench_aligned = bench_oos.loc[start_strat:end_strat]
            top_net       = bt_top.cropped_portfolio_net_returns.loc[start_strat:end_strat]
            bot_net       = bt_bot.cropped_portfolio_net_returns.loc[start_strat:end_strat]

            # ── Performance analysis ─────────────────────────────
            perf_top = _make_perf(top_net, bench_aligned, config, rolling_win)
            perf_bot = _make_perf(bot_net, bench_aligned, config, rolling_win)

            _print_feature_summary(feature_name, threshold, perf_top, perf_bot, len(top_net))

            # ── Accumulate for summary table ─────────────────────
            m_t = perf_top.metrics
            m_b = perf_bot.metrics
            summary_rows.append({
                "Strategy": f"{feature_name} (top)",
                "TotRet": m_t["total_return"],
                "AnnRet": m_t["annualized_return"],
                "Vol":    m_t["annualized_volatility"],
                "SR":     m_t["annualized_sharpe_ratio"],
                "MaxDD":  m_t["max_drawdown"],
                "is_bench": False,
            })
            summary_rows.append({
                "Strategy": f"{feature_name} (bottom)",
                "TotRet": m_b["total_return"],
                "AnnRet": m_b["annualized_return"],
                "Vol":    m_b["annualized_volatility"],
                "SR":     m_b["annualized_sharpe_ratio"],
                "MaxDD":  m_b["max_drawdown"],
                "is_bench": False,
            })
            bench_row = {
                "Strategy": f"Buy & Hold ({mkt_col})",
                "TotRet": _scalar(m_t["total_return_bench"]),
                "AnnRet": _scalar(m_t["annualized_return_bench"]),
                "Vol":    _scalar(m_t["annualized_volatility_bench"]),
                "SR":     _scalar(m_t["annualized_sharpe_ratio_bench"]),
                "MaxDD":  _scalar(m_t["max_drawdown_bench"]),
                "is_bench": True,
            }

            # ── Per-feature plots ────────────────────────────────
            feat_dir = output_dir / feature_name / f"threshold_{thr_str}"
            feat_dir.mkdir(parents=True, exist_ok=True)

            # Cumulative: both directions + bench on one chart
            _plot_feature_cumulative(
                perf_top, perf_bot, feature_name, threshold,
                feat_dir / "cumulative_returns.png",
            )

            # Rolling metrics: separate plot per direction
            vizu_top = Visualizer(performance=perf_top)
            vizu_bot = Visualizer(performance=perf_bot)
            for metric in ["sharpe", "return", "vol"]:
                vizu_top.plot_rolling_metric(
                    metric=metric, window=rolling_win,
                    saving_path=feat_dir / f"rolling_{metric}_top.png",
                )
                vizu_bot.plot_rolling_metric(
                    metric=metric, window=rolling_win,
                    saving_path=feat_dir / f"rolling_{metric}_bottom.png",
                )

            # Yearly metrics: separate plot per direction
            for metric in ["annualized_return", "vol", "sharpe"]:
                vizu_top.plot_yearly_metrics(
                    metric=metric, show=False,
                    saving_path=feat_dir / f"yearly_{metric}_top.png",
                )
                vizu_bot.plot_yearly_metrics(
                    metric=metric, show=False,
                    saving_path=feat_dir / f"yearly_{metric}_bottom.png",
                )

            logger.info(
                "Feature '%s', threshold %s — figures saved to %s",
                feature_name, threshold, feat_dir,
            )

            # ── Collect for combined chart ───────────────────────
            all_cum_perfs[f"{feature_name} (top)"]    = perf_top.cumulative_performance_base_100.iloc[:, 0]
            all_cum_perfs[f"{feature_name} (bottom)"] = perf_bot.cumulative_performance_base_100.iloc[:, 0]
            if bench_cum_ref is None:
                bench_cum_ref = perf_top.bench_cumulative_perf_base_100

        # ── Combined cumulative chart for this threshold ─────────
        if all_cum_perfs and bench_cum_ref is not None:
            combined_dir = output_dir / "combined" / f"threshold_{thr_str}"
            combined_dir.mkdir(parents=True, exist_ok=True)
            _plot_combined_cumulative(
                all_cum_perfs=all_cum_perfs,
                bench_cum=bench_cum_ref,
                threshold=threshold,
                feature_names=config.cs_backtest_node_features,
                output_path=combined_dir / "combined_cumulative.png",
            )
            logger.info(
                "Combined chart for threshold %s saved to %s", threshold, combined_dir,
            )

        # ── Summary table for this threshold ─────────────────────
        if summary_rows:
            if bench_row is not None:
                summary_rows.append(bench_row)
            _print_summary_table(summary_rows, threshold)

    logger.info("Cross-sectional backtest complete. Figures saved to %s", output_dir)
