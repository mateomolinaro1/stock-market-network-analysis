"""
Oracle (perfect-foresight) market-timing backtest.

Callable from main.py via run_oracle_timing_backtest(), or from the
standalone script at scripts/timing_backtest.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from stock_mkt_network_analysis.backtester.analysis import PerformanceAnalyser
from stock_mkt_network_analysis.backtester.backtest import Backtest
from stock_mkt_network_analysis.backtester.visualization import Visualizer
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.utils.config import Config

_TRANSACTION_COSTS_BPS = 10   # per leg; a regime switch touches 2 legs → 20 bps


def run_oracle_timing_backtest(
    config: Config,
    data_manager: DataManager,
    output_dir: Path,
) -> None:
    """
    Perfect-foresight market-timing backtest.

    Invests in the market (Russell 1000) when the forward max-drawdown
    regime is 0, and switches to the risk-free asset for 21 days whenever
    the oracle target fires (regime = 1).

    Expects data_manager.load_data() to have been called beforehand.
    Saves performance plots to output_dir.
    """
    logger = logging.getLogger(__name__)

    # ── OOS start (consistent with CV burn-in) ─────────────────
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
        "Oracle backtest — OOS window: %s → %s  (%d trading days)",
        start_oos.date(), dates[-1].date(), len(dates) - start_oos_loc,
    )

    # ── Oracle regime signal ────────────────────────────────────
    # target[t] = 1 → crisis over [t+1, t+21]: hold RF for those 21 days.
    # holding_rf[τ] = max(target[τ-21], …, target[τ-1])
    #   shift(1)      → oracle issued at end-of-day t arrives at τ = t+1
    #   rolling(21)   → stay in RF for the full announced crisis window
    target = data_manager.target_variable_to_predict.squeeze().dropna()

    holding_rf = (
        target
        .shift(1)
        .rolling(window=config.target_variable_rolling_window, min_periods=1)
        .max()
        .fillna(0)
        .clip(0, 1)
        .astype(int)
    )
    holding_rf_oos = holding_rf.loc[start_oos:]

    # ── Returns for the two legs ────────────────────────────────
    mkt_col = config.mkt_benchmark_column

    mkt = data_manager.mkt_returns[[mkt_col]].reindex(holding_rf_oos.index)

    rf_series = (
        data_manager.rf_returns.iloc[:, 0]
        .reindex(holding_rf_oos.index)
        .ffill(limit=config.limit_ffill_rf)
        .to_frame("rf")
    )

    combined_returns = pd.concat([mkt, rf_series], axis=1)
    combined_returns.columns = ["market", "rf"]
    combined_returns = combined_returns.dropna()

    holding_rf_aligned = (
        holding_rf_oos
        .reindex(combined_returns.index)
        .fillna(0)
        .astype(int)
    )

    # ── Portfolio weights  [market, rf] ────────────────────────
    weights = pd.DataFrame(
        {
            "market": (1 - holding_rf_aligned).values,
            "rf":     holding_rf_aligned.values,
        },
        index=combined_returns.index,
        dtype=float,
    )

    # Turnover: a regime switch flips both legs → |Δw| summed = 2.
    turnover = weights.diff().abs().sum(axis=1).to_frame("turnover")
    turnover.iloc[0] = weights.iloc[0].abs().sum()   # initial build cost

    # ── Backtest ────────────────────────────────────────────────
    backtester = Backtest(
        returns=combined_returns,
        weights=weights,
        turnover=turnover,
        transaction_costs=_TRANSACTION_COSTS_BPS,
        strategy_name="perfect_foresight_timing",
    )
    backtester.run_backtest()

    # ── Buy-and-hold benchmark ──────────────────────────────────
    bench_returns = (
        mkt
        .reindex(backtester.cropped_portfolio_net_returns.index)
        .rename(columns={"market": mkt_col})
    )

    # ── Performance analysis ────────────────────────────────────
    perf = PerformanceAnalyser(
        portfolio_returns=backtester.cropped_portfolio_net_returns,
        freq=config.data_freq,
        bench_returns=bench_returns,
        rebal_freq="on regime switch",
    )
    perf.compute_metrics()
    perf.compute_rolling_metrics(window=252)
    perf.compute_yearly_metrics()

    # ── Terminal output ─────────────────────────────────────────
    m = perf.metrics

    def _b(key):
        val = m[key]
        return val.iloc[0] if hasattr(val, "iloc") else val

    n_oos      = len(holding_rf_aligned)
    n_rf_days  = int(holding_rf_aligned.sum())
    n_switches = int((holding_rf_aligned.diff().abs() == 1).sum())

    print()
    print("=" * 60)
    print("  PERFECT-FORESIGHT MARKET TIMING — OOS METRICS")
    print("=" * 60)
    print(f"  {'Metric':<38} {'Strategy':>9}  {'B&H':>9}")
    print("-" * 60)
    print(f"  {'Total Return':<38} {m['total_return']:>9.2%}  {_b('total_return_bench'):>9.2%}")
    print(f"  {'Annualized Return':<38} {m['annualized_return']:>9.2%}  {_b('annualized_return_bench'):>9.2%}")
    print(f"  {'Annualized Volatility':<38} {m['annualized_volatility']:>9.2%}  {_b('annualized_volatility_bench'):>9.2%}")
    print(f"  {'Sharpe Ratio':<38} {m['annualized_sharpe_ratio']:>9.2f}  {_b('annualized_sharpe_ratio_bench'):>9.2f}")
    print(f"  {'Max Drawdown':<38} {m['max_drawdown']:>9.2%}  {_b('max_drawdown_bench'):>9.2%}")
    print("=" * 60)
    print(
        f"\n  OOS period : {backtester.start_date.date()} → "
        f"{backtester.cropped_portfolio_net_returns.index[-1].date()} "
        f"({n_oos} trading days)"
    )
    print(f"  Days in RF : {n_rf_days} / {n_oos}  ({n_rf_days / n_oos:.1%})")
    print(f"  Days in mkt: {n_oos - n_rf_days} / {n_oos}  ({(n_oos - n_rf_days) / n_oos:.1%})")
    print(
        f"  Switches   : {n_switches}  "
        f"({_TRANSACTION_COSTS_BPS} bps × 2 legs = {2 * _TRANSACTION_COSTS_BPS} bps per switch)\n"
    )

    # ── Plots ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    perf.compute_cumulative_performance()
    strat_cum = perf.cumulative_performance_base_100
    bench_cum = perf.bench_cumulative_perf_base_100
    regime    = holding_rf_aligned.reindex(strat_cum.index).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))

    in_crisis    = False
    crisis_start = None
    for date, regime_val in regime.items():
        if regime_val == 1 and not in_crisis:
            crisis_start = date
            in_crisis = True
        elif regime_val == 0 and in_crisis:
            ax.axvspan(crisis_start, date, color="salmon", alpha=0.25, linewidth=0)
            in_crisis = False
    if in_crisis:
        ax.axvspan(crisis_start, regime.index[-1], color="salmon", alpha=0.25, linewidth=0)

    ax.plot(strat_cum.index, strat_cum[strat_cum.columns[0]],
            label="Perfect foresight timing", linewidth=1.5)
    for col in bench_cum.columns:
        ax.plot(bench_cum.index, bench_cum[col],
                label=f"Buy & Hold — {col}", linestyle="--", linewidth=1.5)

    rf_patch = mpatches.Patch(color="salmon", alpha=0.4, label="Risk-free regime (oracle = 1)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [rf_patch], fontsize=9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance (Base 100)")
    ax.grid(True, alpha=0.4)
    ax.set_title(
        f"Perfect-Foresight Market Timing — Cumulative Performance (Base 100)\n"
        f"Strategy: ann.ret={m['annualized_return']:.2%}, "
        f"vol={m['annualized_volatility']:.2%}, "
        f"SR={m['annualized_sharpe_ratio']:.2f}, "
        f"maxDD={m['max_drawdown']:.2%}  |  "
        f"Days in RF: {n_rf_days}/{n_oos} ({n_rf_days/n_oos:.1%})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    vizu = Visualizer(performance=perf)
    for metric in ["sharpe", "return", "vol"]:
        vizu.plot_rolling_metric(
            metric=metric,
            window=252,
            saving_path=output_dir / f"rolling_{metric}.png",
        )
    for metric in ["annualized_return", "vol", "sharpe"]:
        vizu.plot_yearly_metrics(
            metric=metric,
            saving_path=output_dir / f"yearly_{metric}.png",
        )

    logger.info("Oracle timing backtest complete. Figures saved to %s", output_dir)