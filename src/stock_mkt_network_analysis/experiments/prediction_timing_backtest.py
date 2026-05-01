"""
Prediction-based market-timing backtest.

For each (model, feature_mode) pair from the walk-forward CV, converts
y_pred_binary into a market-timing signal that mirrors the oracle:
  - y_pred_binary[t] = 1 → hold risk-free for the next TARGET_VARIABLE_ROLLING_WINDOW days
  - y_pred_binary[t] = 0 → hold market

The signal is shifted by 1 day (end-of-day t prediction → trade from t+1)
and extended with a rolling max to capture the full announced crisis window.

Saves per-(model, feature_mode) plots to:
    output_dir / [model]__[feature_mode] / *.png
Combined cumulative chart (all strategies + benchmark):
    output_dir / combined_cumulative.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from stock_mkt_network_analysis.backtester.analysis import PerformanceAnalyser
from stock_mkt_network_analysis.backtester.backtest import Backtest
from stock_mkt_network_analysis.backtester.visualization import Visualizer
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.utils.config import Config

_TRANSACTION_COSTS_BPS = 10


# ── Helpers ────────────────────────────────────────────────────────────────


def _b(metrics: dict, key: str) -> float:
    val = metrics[key]
    return float(val.iloc[0]) if hasattr(val, "iloc") else float(val)


def _build_timing_backtest(
    pred_binary: pd.Series,
    config: Config,
    data_manager: DataManager,
    strategy_name: str,
) -> Tuple[Backtest, pd.Series]:
    """
    Convert a date-indexed binary prediction series into a market-timing backtest.

    Returns (Backtest, holding_rf_aligned) where holding_rf_aligned is the
    series of 0/1 regime flags aligned to the backtest date range.
    """
    mkt_col = config.mkt_benchmark_column

    # Shift by 1: prediction issued at end of t → effective from t+1
    # Rolling max: stay in RF for the full target_variable_rolling_window window
    holding_rf = (
        pred_binary
        .shift(1)
        .rolling(window=config.target_variable_rolling_window, min_periods=1)
        .max()
        .fillna(0)
        .clip(0, 1)
        .astype(int)
    )

    mkt = data_manager.mkt_returns[[mkt_col]].reindex(holding_rf.index)
    rf = (
        data_manager.rf_returns.iloc[:, 0]
        .reindex(holding_rf.index)
        .ffill(limit=config.limit_ffill_rf)
        .to_frame("rf")
    )

    combined_returns = pd.concat([mkt, rf], axis=1)
    combined_returns.columns = ["market", "rf"]
    combined_returns = combined_returns.dropna()

    holding_rf_aligned = (
        holding_rf
        .reindex(combined_returns.index)
        .fillna(0)
        .astype(int)
    )

    weights = pd.DataFrame(
        {
            "market": (1 - holding_rf_aligned).values,
            "rf":     holding_rf_aligned.values,
        },
        index=combined_returns.index,
        dtype=float,
    )

    turnover = weights.diff().abs().sum(axis=1).to_frame("turnover")
    turnover.iloc[0] = weights.iloc[0].abs().sum()

    bt = Backtest(
        returns=combined_returns,
        weights=weights,
        turnover=turnover,
        transaction_costs=_TRANSACTION_COSTS_BPS,
        strategy_name=strategy_name,
    )
    bt.run_backtest()
    return bt, holding_rf_aligned


# ── Terminal output ─────────────────────────────────────────────────────────


def _print_strategy_summary(
    strategy_label: str,
    bt: Backtest,
    perf: PerformanceAnalyser,
    holding_rf_aligned: pd.Series,
) -> None:
    m = perf.metrics
    n_oos      = len(holding_rf_aligned)
    n_rf_days  = int(holding_rf_aligned.sum())
    n_switches = int((holding_rf_aligned.diff().abs() == 1).sum())

    print()
    print("=" * 60)
    print(f"  PREDICTION TIMING — {strategy_label}")
    print("=" * 60)
    print(f"  {'Metric':<38} {'Strategy':>9}  {'B&H':>9}")
    print("-" * 60)
    print(f"  {'Total Return':<38} {m['total_return']:>9.2%}  {_b(m, 'total_return_bench'):>9.2%}")
    print(f"  {'Annualized Return':<38} {m['annualized_return']:>9.2%}  {_b(m, 'annualized_return_bench'):>9.2%}")
    print(f"  {'Annualized Volatility':<38} {m['annualized_volatility']:>9.2%}  {_b(m, 'annualized_volatility_bench'):>9.2%}")
    print(f"  {'Sharpe Ratio':<38} {m['annualized_sharpe_ratio']:>9.2f}  {_b(m, 'annualized_sharpe_ratio_bench'):>9.2f}")
    print(f"  {'Max Drawdown':<38} {m['max_drawdown']:>9.2%}  {_b(m, 'max_drawdown_bench'):>9.2%}")
    print("=" * 60)
    print(
        f"\n  OOS period : {bt.start_date.date()} → "
        f"{bt.cropped_portfolio_net_returns.index[-1].date()} "
        f"({len(bt.cropped_portfolio_net_returns)} trading days)"
    )
    print(f"  Days in RF : {n_rf_days} / {n_oos}  ({n_rf_days / n_oos:.1%})")
    print(f"  Days in mkt: {n_oos - n_rf_days} / {n_oos}  ({(n_oos - n_rf_days) / n_oos:.1%})")
    print(
        f"  Switches   : {n_switches}  "
        f"({_TRANSACTION_COSTS_BPS} bps × 2 legs = {2 * _TRANSACTION_COSTS_BPS} bps per switch)\n"
    )


def _print_summary_table(summary_rows: list, mkt_col: str) -> None:
    if not summary_rows:
        return
    sorted_rows = sorted(summary_rows, key=lambda r: r["SR"], reverse=True)
    W = 84
    print()
    print("=" * W)
    print("  PREDICTION TIMING SUMMARY  (sorted by Sharpe Ratio ↓)")
    print("=" * W)
    print(f"  {'Strategy':<36} {'TotRet':>8}  {'AnnRet':>8}  {'Vol':>8}  {'SR':>6}  {'MaxDD':>9}  {'%inRF':>6}")
    print("-" * W)
    for r in sorted_rows:
        tag  = " ★" if r.get("is_bench") else ""
        days = f"{r['DaysRF%']:.1%}" if pd.notna(r["DaysRF%"]) else "   n/a"
        print(
            f"  {r['Strategy'] + tag:<36} {r['TotRet']:>8.2%}  {r['AnnRet']:>8.2%}  "
            f"{r['Vol']:>8.2%}  {r['SR']:>6.2f}  {r['MaxDD']:>9.2%}  {days:>6}"
        )
    print("=" * W)
    print("  ★ = benchmark")


# ── Plots ───────────────────────────────────────────────────────────────────


def _plot_cumulative(
    bt: Backtest,
    perf: PerformanceAnalyser,
    holding_rf: pd.Series,
    strategy_label: str,
    n_rf_days: int,
    n_oos: int,
    n_switches: int,
    output_path: Path,
) -> None:
    m = perf.metrics

    strat_cum = perf.cumulative_performance_base_100
    bench_cum = perf.bench_cumulative_perf_base_100
    regime    = holding_rf.reindex(strat_cum.index).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))

    in_crisis    = False
    crisis_start = None
    for date, rv in regime.items():
        if rv == 1 and not in_crisis:
            crisis_start = date
            in_crisis = True
        elif rv == 0 and in_crisis:
            ax.axvspan(crisis_start, date, color="salmon", alpha=0.25, linewidth=0)
            in_crisis = False
    if in_crisis:
        ax.axvspan(crisis_start, regime.index[-1], color="salmon", alpha=0.25, linewidth=0)

    ax.plot(strat_cum.index, strat_cum.iloc[:, 0],
            label=strategy_label, linewidth=1.5)
    for col in bench_cum.columns:
        ax.plot(bench_cum.index, bench_cum[col],
                label=f"Buy & Hold — {col}", linestyle="--", linewidth=1.5)

    rf_patch = mpatches.Patch(color="salmon", alpha=0.4, label="Risk-free regime (predicted = 1)")
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [rf_patch], fontsize=9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance (Base 100)")
    ax.grid(True, alpha=0.4)
    ax.set_title(
        f"Prediction-Based Market Timing — {strategy_label}\n"
        f"Strategy: ann.ret={m['annualized_return']:.2%}, vol={m['annualized_volatility']:.2%}, "
        f"SR={m['annualized_sharpe_ratio']:.2f}, maxDD={m['max_drawdown']:.2%}  |  "
        f"Days in RF: {n_rf_days}/{n_oos} ({n_rf_days/n_oos:.1%})  |  Switches: {n_switches}\n"
        f"Benchmark:  ann.ret={_b(m, 'annualized_return_bench'):.2%}, "
        f"vol={_b(m, 'annualized_volatility_bench'):.2%}, "
        f"SR={_b(m, 'annualized_sharpe_ratio_bench'):.2f}, "
        f"maxDD={_b(m, 'max_drawdown_bench'):.2%}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_combined_cumulative(
    cum_series: Dict[str, pd.Series],
    bench_cum: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    for label, series in cum_series.items():
        ax.plot(series.index, series.values, label=label, linewidth=1.5)
    for col in bench_cum.columns:
        ax.plot(bench_cum.index, bench_cum[col],
                label=f"Buy & Hold — {col}", linestyle="--", linewidth=2.0, color="black")

    ax.set_title(
        "Prediction-Based Market Timing — All Models\nCumulative Performance (Base 100)",
        fontsize=11,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance (Base 100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Main entry point ────────────────────────────────────────────────────────


def run_prediction_timing_backtest(
    config: Config,
    data_manager: DataManager,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Prediction-based market-timing backtest.

    predictions must be the flat DataFrame returned by run_cv() with columns:
        date, model, feature_mode, y_true, y_pred, y_pred_binary,
        best_threshold, validation_score

    For each (model, feature_mode) pair, y_pred_binary drives a market-timing
    strategy identical in structure to the oracle (Step 5):
      - predicted regime = 1 → hold risk-free for TARGET_VARIABLE_ROLLING_WINDOW days
      - predicted regime = 0 → hold market

    Expects data_manager.load_data() to have been called beforehand.
    Saves performance plots to output_dir.
    """
    logger = logging.getLogger(__name__)

    if predictions is None or predictions.empty:
        logger.warning("predictions is empty; skipping prediction timing backtest.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    mkt_col = config.mkt_benchmark_column

    all_cum_series: Dict[str, pd.Series] = {}
    bench_cum_ref:  Optional[pd.DataFrame] = None
    summary_rows:   list = []
    last_perf:      Optional[PerformanceAnalyser] = None

    groups = predictions.groupby(["model", "feature_mode"])

    for (model_name, feature_mode), grp in groups:
        strategy_label = f"{model_name} | {feature_mode}"
        strat_slug     = f"{model_name}__{feature_mode}"
        logger.info("Prediction timing backtest — %s", strategy_label)

        # ── Binary prediction series indexed by date ────────────
        pred_binary = (
            grp.set_index("date")["y_pred_binary"]
            .sort_index()
            .fillna(0)
            .astype(int)
        )
        if pred_binary.empty:
            logger.warning("No predictions for %s — skipping.", strategy_label)
            continue

        # ── Build and run backtest ───────────────────────────────
        bt, holding_rf_aligned = _build_timing_backtest(
            pred_binary=pred_binary,
            config=config,
            data_manager=data_manager,
            strategy_name=strat_slug,
        )

        bench_returns = (
            data_manager.mkt_returns[[mkt_col]]
            .reindex(bt.cropped_portfolio_net_returns.index)
        )

        # ── Performance analysis ─────────────────────────────────
        perf = PerformanceAnalyser(
            portfolio_returns=bt.cropped_portfolio_net_returns,
            bench_returns=bench_returns,
            freq=config.data_freq,
            rebal_freq="on regime switch",
        )
        perf.compute_cumulative_performance()
        perf.compute_metrics()
        perf.compute_rolling_metrics(window=config.cs_backtest_rolling_window_metrics)
        perf.compute_yearly_metrics()
        last_perf = perf

        n_oos      = len(holding_rf_aligned)
        n_rf_days  = int(holding_rf_aligned.sum())
        n_switches = int((holding_rf_aligned.diff().abs() == 1).sum())

        _print_strategy_summary(strategy_label, bt, perf, holding_rf_aligned)

        # ── Save per-strategy plots ──────────────────────────────
        strat_dir = output_dir / strat_slug
        strat_dir.mkdir(parents=True, exist_ok=True)

        _plot_cumulative(
            bt=bt,
            perf=perf,
            holding_rf=holding_rf_aligned,
            strategy_label=strategy_label,
            n_rf_days=n_rf_days,
            n_oos=n_oos,
            n_switches=n_switches,
            output_path=strat_dir / "cumulative_returns.png",
        )

        vizu = Visualizer(performance=perf)
        for metric in ["sharpe", "return", "vol"]:
            vizu.plot_rolling_metric(
                metric=metric,
                window=config.cs_backtest_rolling_window_metrics,
                saving_path=strat_dir / f"rolling_{metric}.png",
            )
        for metric in ["annualized_return", "vol", "sharpe"]:
            vizu.plot_yearly_metrics(
                metric=metric, show=False,
                saving_path=strat_dir / f"yearly_{metric}.png",
            )

        logger.info("Plots saved to %s", strat_dir)

        # ── Collect for combined chart and summary table ─────────
        all_cum_series[strategy_label] = perf.cumulative_performance_base_100.iloc[:, 0]
        if bench_cum_ref is None:
            bench_cum_ref = perf.bench_cumulative_perf_base_100

        m = perf.metrics
        summary_rows.append({
            "Strategy": strategy_label,
            "TotRet":   m["total_return"],
            "AnnRet":   m["annualized_return"],
            "Vol":      m["annualized_volatility"],
            "SR":       m["annualized_sharpe_ratio"],
            "MaxDD":    m["max_drawdown"],
            "DaysRF%":  n_rf_days / n_oos if n_oos > 0 else float("nan"),
            "is_bench": False,
        })

    # ── Combined cumulative chart ────────────────────────────────
    if all_cum_series and bench_cum_ref is not None:
        _plot_combined_cumulative(
            cum_series=all_cum_series,
            bench_cum=bench_cum_ref,
            output_path=output_dir / "combined_cumulative.png",
        )
        logger.info("Combined chart saved to %s", output_dir / "combined_cumulative.png")

    # ── Summary table ────────────────────────────────────────────
    if summary_rows and last_perf is not None:
        m_last = last_perf.metrics
        summary_rows.append({
            "Strategy": f"Buy & Hold ({mkt_col})",
            "TotRet":   _b(m_last, "total_return_bench"),
            "AnnRet":   _b(m_last, "annualized_return_bench"),
            "Vol":      _b(m_last, "annualized_volatility_bench"),
            "SR":       _b(m_last, "annualized_sharpe_ratio_bench"),
            "MaxDD":    _b(m_last, "max_drawdown_bench"),
            "DaysRF%":  float("nan"),
            "is_bench": True,
        })
        _print_summary_table(summary_rows, mkt_col)

    logger.info("Prediction timing backtest complete. Figures saved to %s", output_dir)
