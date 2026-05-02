# US Stock Market Network Analysis for Crisis Prediction

A research pipeline that models the Russell 1000 equity universe as a dynamic correlation network and tests whether its topology can predict market crises. The project combines robust correlation estimation (Ledoit–Wolf shrinkage + exponential time-weighting), threshold graph construction, walk-forward cross-validation, and three backtests: oracle timing, prediction-based timing, and cross-sectional stock selection from node-level network features.

A full write-up is available at `outputs/reports/report_network_analysis.tex`.

---

## Table of contents

1. [Setup](#setup)
2. [Configuration](#configuration)
3. [Running the pipeline](#running-the-pipeline)
4. [Pipeline steps](#pipeline-steps)
5. [Outputs](#outputs)
6. [Project structure](#project-structure)

---

## Setup

### 1. Navigate to the folder where you want to clone the project

```bash
cd /path/to/your/desired/ROOT
```

### 2. Clone the repository

```bash
git clone https://github.com/mateomolinaro1/stock-market-network-analysis.git
cd stock-market-network-analysis
```

### 3. Create a `.env` file

Create a `.env` file at the root of the repository with your AWS credentials (data is stored in S3):

```env
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key
```

### 4. Install `uv` (if not already installed)

```bash
pip show uv || pip install uv
```

### 5. Create a virtual environment

```bash
uv venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 6. Install dependencies

```bash
uv sync
```

---

## Configuration

All pipeline parameters live in `config/run_pipeline_config.json`. Key sections:

| Section | Key parameters |
|---|---|
| `AWS.S3` | S3 bucket name, region, filenames to load |
| `DATA` | Date range, target variable, rolling window (21d), crisis quantile (0.20) |
| `FORECASTING` | Lookback (252d), train/val sizes, feature modes (`ts`, `network`, `all`), model grid, cache flags |
| `BACKTEST` | Transaction costs (10 bps), portfolio percentiles (20/80), node features to test |

The `LOAD_OR_COMPUTE_*` / `SAVE_*` flags in `FORECASTING` control whether correlation matrices, time-series features, graph features, and node features are recomputed or loaded from S3 cache on each run. Set them to `"load"` after the first run to skip expensive recomputation.

---

## Running the pipeline

```bash
python main.py
```

The full pipeline takes several hours on first run (computing correlation matrices and feature caches for the full Russell 1000 universe). Subsequent runs with `LOAD_OR_COMPUTE_*: "load"` are fast.

To run only the timing backtest script independently:

```bash
python scripts/timing_backtest.py
```

---

## Pipeline steps

The pipeline runs 7 steps tracked by a progress bar:

| Step | Description |
|---|---|
| 1 | **Load data** — fetch Russell 1000 returns, market index, and risk-free rate from S3 |
| 2 | **Precompute caches** — rolling Ledoit–Wolf correlation matrices, time-series features, graph-level features, node-level features |
| 3 | **Walk-forward CV** — expanding-window cross-validation across 3 feature modes (`ts`, `network`, `all`) and 3 classifiers (Logistic Regression, Random Forest, Gradient Boosting); optimise PR-AUC |
| 4 | *(Analytics — optional, commented out in default config)* |
| 5 | **Oracle timing backtest** — perfect-foresight upper bound: hold market when target = 0, hold risk-free for 21 days when target = 1 |
| 6 | **Prediction timing backtest** — same structure as oracle but driven by walk-forward binary predictions; one strategy per (model, feature mode) pair |
| 7 | **Cross-sectional backtest** — use node-level features (degree, strength, clustering, eigenvector centrality, PageRank, core number) as cross-sectional signals; long-top and long-bottom equal-weight portfolios with 1-day implementation lag |

---

## Outputs

```
outputs/
├── figures/
│   ├── backtest_oracle_timing/          # Cumulative returns, rolling metrics, yearly breakdown
│   ├── backtest_prediction_timing/
│   │   ├── <Model>__<feature_mode>/     # Per-strategy charts
│   │   └── combined_cumulative_prediction_timing.png
│   ├── backtest_cross_sectional/
│   │   ├── <feature>/threshold_0_75/   # Per-feature long-top + long-bottom charts
│   │   └── combined/threshold_0_75/combined_cumulative.png
│   └── ...                              # Target variable and network analytics plots
├── reports/
│   └── report_network_analysis.tex     # Full LaTeX report with results and interpretation
├── results/
└── node_features_threshold_0.75.parquet    # Cached node-level feature panel
```

### Key results (threshold τ = 0.75)

**Predictive performance** — all PR-AUCs are in the range 0.22–0.29, modestly above the 0.20 naive baseline. Time-series features consistently outperform network-only features. No classifier comes close to the oracle timing benchmark.

**Cross-sectional backtest** — the strongest result: long-periphery portfolios (lowest eigenvector centrality or clustering) achieve Sharpe ratios of 1.14 and 1.00 respectively, above the Russell 1000 buy-and-hold Sharpe of 0.89, with maximum drawdowns of −15% vs −35% for the benchmark.

---

## Project structure

```
.
├── config/
│   └── run_pipeline_config.json        # All pipeline parameters
├── src/stock_mkt_network_analysis/
│   ├── analytics/                      # Network analytics and visualizations
│   ├── backtester/                     # Backtest engine, strategies, portfolio, performance
│   ├── cv/                             # Walk-forward CV, feature pipeline
│   ├── data/                           # DataManager (S3 loading, alignment)
│   ├── experiments/                    # Runnable experiment modules
│   │   ├── run_wf_cv.py
│   │   ├── timing_backtest.py          # Oracle timing
│   │   ├── prediction_timing_backtest.py
│   │   └── cross_sectional_backtest.py
│   ├── network/                        # Correlation estimation, graph builder, feature extraction
│   ├── time_series/                    # Adaptive time-series feature extractor
│   └── utils/                          # Config, caching layers, metrics
├── scripts/                            # Standalone scripts
├── outputs/                            # All generated artifacts (see above)
├── main.py                             # Full pipeline entry point
└── pyproject.toml
```
