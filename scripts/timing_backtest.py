"""
timing_backtest.py — Perfect-foresight market-timing backtest.

Strategy: invest in the Russell 1000 when the forward max-drawdown regime
signal is 0 (normal), switch to the risk-free asset when it is 1 (crisis).

"Perfect foresight" means the regime label at time t was computed from the
realised market drawdown over the *next* 21 trading days, so this script
establishes a performance upper-bound rather than a live strategy.

Signal mechanics
----------------
  target[t] = 1 → oracle knows the next 21 days will be a crisis.
              We enter the risk-free asset at t+1 and hold for 21 days.

  holding_rf[τ] = max(target[τ-21], …, target[τ-1])

  This is 1 on exactly the days that fall inside any crisis window opened
  by a past oracle signal, and 0 otherwise.  It uses no information beyond
  what was available at the end of day τ-1 (the "past" oracles are
  forward-looking by construction, but they were issued in the past).

  Rebalancing occurs only when holding_rf changes (low turnover).

Backtester usage
----------------
  - Backtest          → portfolio gross/net return series
  - PerformanceAnalyser → metrics
  - Visualizer        → saved plots
  EqualWeightingScheme is bypassed: weights are built directly for a
  2-asset portfolio [market, rf].
"""

from dotenv import load_dotenv
import logging
import sys

import pandas as pd
import numpy as np

from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.backtester.backtest import Backtest
from stock_mkt_network_analysis.backtester.analysis import PerformanceAnalyser
from stock_mkt_network_analysis.backtester.visualization import Visualizer

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
TRANSACTION_COSTS_BPS = 10   # bps per leg; a regime switch touches 2 legs → 20 bps

# ──────────────────────────────────────────────────────────────
# Logging / env
# ──────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
config = Config()
data_manager = DataManager(config=config)
data_manager.load_data()

# ──────────────────────────────────────────────────────────────
# OOS start date (consistent with CV burn-in)
# ──────────────────────────────────────────────────────────────
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
logger.info("OOS window: %s → %s  (%d trading days)",
            start_oos.date(), dates[-1].date(), len(dates) - start_oos_loc)

# ──────────────────────────────────────────────────────────────
# Oracle regime signal
# ──────────────────────────────────────────────────────────────
# target[t] = 1 means the next `rolling_window` days are a crisis.
# Shift by 1 so that the oracle issued at end-of-day t arrives at day t+1,
# then take a rolling max over the full crisis window so we stay in RF for
# all 21 days that any past oracle signal announced as a crisis.
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

# Restrict to OOS
holding_rf_oos = holding_rf.loc[start_oos:]

# ──────────────────────────────────────────────────────────────
# Returns for the two legs
# ──────────────────────────────────────────────────────────────
mkt_col = config.mkt_benchmark_column   # e.g. "Russell 1000"

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

# Align holding signal to the dates where both legs have valid returns
holding_rf_aligned = (
    holding_rf_oos
    .reindex(combined_returns.index)
    .fillna(0)
    .astype(int)
)

# ──────────────────────────────────────────────────────────────
# Portfolio weights  [market, rf]
# ──────────────────────────────────────────────────────────────
weights = pd.DataFrame(
    {
        "market": (1 - holding_rf_aligned).values,
        "rf":     holding_rf_aligned.values,
    },
    index=combined_returns.index,
    dtype=float,
)

# Turnover: a regime switch flips both legs → |Δw_market| + |Δw_rf| = 2.
# First row: cost of building the initial portfolio from zero.
turnover = weights.diff().abs().sum(axis=1).to_frame("turnover")
turnover.iloc[0] = weights.iloc[0].abs().sum()

# ──────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────
backtester = Backtest(
    returns=combined_returns,
    weights=weights,
    turnover=turnover,
    transaction_costs=TRANSACTION_COSTS_BPS,
    strategy_name="perfect_foresight_timing",
)
backtester.run_backtest()

# ──────────────────────────────────────────────────────────────
# Benchmark: buy-and-hold market over the same OOS window
# ──────────────────────────────────────────────────────────────
bench_returns = (
    mkt
    .reindex(backtester.cropped_portfolio_net_returns.index)
    .rename(columns={"market": mkt_col})
)

# ──────────────────────────────────────────────────────────────
# Performance analysis
# ──────────────────────────────────────────────────────────────
perf = PerformanceAnalyser(
    portfolio_returns=backtester.cropped_portfolio_net_returns,
    freq=config.data_freq,
    bench_returns=bench_returns,
    rebal_freq="on regime switch",
)
perf.compute_metrics()
perf.compute_rolling_metrics(window=252)
perf.compute_yearly_metrics()

# ──────────────────────────────────────────────────────────────
# Terminal metrics
# ──────────────────────────────────────────────────────────────
m = perf.metrics

def _b(key):
    """First element of a bench metric Series."""
    val = m[key]
    return val.iloc[0] if hasattr(val, "iloc") else val

n_oos       = len(holding_rf_aligned)
n_rf_days   = int(holding_rf_aligned.sum())
n_switches  = int((holding_rf_aligned.diff().abs() == 1).sum())

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
print(f"\n  OOS period : {backtester.start_date.date()} → "
      f"{backtester.cropped_portfolio_net_returns.index[-1].date()} "
      f"({n_oos} trading days)")
print(f"  Days in RF : {n_rf_days} / {n_oos}  ({n_rf_days / n_oos:.1%})")
print(f"  Days in mkt: {n_oos - n_rf_days} / {n_oos}  ({(n_oos - n_rf_days) / n_oos:.1%})")
print(f"  Switches   : {n_switches}  "
      f"({TRANSACTION_COSTS_BPS} bps × 2 legs = {2 * TRANSACTION_COSTS_BPS} bps per switch)\n")

# ──────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────
output_dir = config.ROOT_DIR / "outputs" / "figures" / "timing_backtest"
output_dir.mkdir(parents=True, exist_ok=True)

vizu = Visualizer(performance=perf)

vizu.plot_cumulative_performance(
    saving_path=output_dir / "cumulative_returns.png",
)
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

logger.info("Timing backtest complete. Figures saved to %s", output_dir)