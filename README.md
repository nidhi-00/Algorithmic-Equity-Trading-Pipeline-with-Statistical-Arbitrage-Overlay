# Algorithmic Equity Trading Pipeline with Statistical Arbitrage Overlay

## Project overview

This repository implements an end-to-end quantitative trading research pipeline for the Precog Quant Task 2026. The task is to transform raw anonymized OHLCV equity data into cleaned features, predictive signals, portfolio weights, realistic transaction-cost-aware backtests, benchmark comparisons, and statistical-arbitrage pair analysis.

The project covers four parts:

1. Feature Engineering & Data Cleaning
2. Model Training & Strategy Formulation
3. Backtesting & Performance Analysis
4. Statistical Arbitrage Overlay

## Final result summary

Final run used the real Kaggle anonymized equity universe locally. The dataset itself is **not committed** to this repository and must be downloaded separately.

- Universe: 100 anonymized assets
- Raw rows: 251,100
- Rows after cleaning: 251,081
- Full data range: 2016-01-25 to 2026-01-16
- Out-of-sample test start: 2024-01-08
- Initial capital: $1,000,000
- Transaction cost assumption: 10 bps per unit traded
- Final strategy: long-only top-ranked portfolio with weekly rebalancing
- Statistical-arbitrage candidates found: 10 pairs

| Metric | Final strategy, with costs | Final strategy, no costs | Equal-weight benchmark |
|---|---:|---:|---:|
| Total return | 84.72% | 107.82% | 47.97% |
| Annualized return | 35.50% | 43.64% | 21.41% |
| Annualized Sharpe | 1.63 | 1.93 | 1.72 |
| Max drawdown | -19.53% | -15.47% | -12.99% |
| Average daily turnover | 23.15% | 23.15% | 0.00% |
| Annualized turnover | 58.33x | 58.33x | 0.00x |
| Final equity | $1,847,178 | $2,078,166 | $1,494,654 |

Interpretation: the final strategy beat the equal-weight benchmark on absolute return and final equity after transaction costs, but it took more active risk, resulting in a larger drawdown and slightly lower Sharpe than the benchmark.

## Resume / QR summary

**Project title:** Transaction-Cost-Aware Equity Alpha and Statistical Arbitrage Research Pipeline

**Resume bullets:**

- Built an end-to-end Python equity alpha research pipeline on 100 anonymized equities and 251k+ OHLCV rows, covering data cleaning, anomaly detection, feature engineering, predictive ranking, transaction-cost-aware backtesting, and statistical-arbitrage pair discovery.
- Developed a walk-forward cross-sectional ranking strategy using Ridge regression and rule-based alpha features; achieved 84.7% net return after 10 bps transaction costs versus 48.0% for an equal-weight benchmark over the out-of-sample test period.
- Evaluated strategy robustness through Sharpe, drawdown, turnover, cost-drag, and benchmark-relative analysis; identified 10 candidate co-moving/cointegrated asset pairs for a potential relative-value overlay.

**One-line interview pitch:**

I built a full quant research pipeline that starts from raw anonymized OHLCV data, engineers tradable cross-sectional features, trains a walk-forward prediction model, converts predictions into portfolio weights, simulates realistic transaction costs, compares against a benchmark, and extends the strategy with statistical-arbitrage pair discovery.

## Repository structure

```text
Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay/
  config/
    default_config.json
    long_only_config.json
  data/
    .gitkeep
  notebooks/
    01_feature_engineering.ipynb
    02_model_and_backtest.ipynb
    03_stat_arb_overlay.ipynb
  outputs/
    .gitkeep
  scripts/
    combine_asset_csvs.py
    run_pipeline.py
  src/precog_trading/
    __init__.py
    backtest.py
    cleaning.py
    features.py
    io.py
    metrics.py
    modelling.py
    pipeline.py
    signals.py
    statarb.py
    synthetic.py
  tests/
    test_backtest.py
    test_cleaning.py
    test_features.py
    test_statarb.py
  workflows/
    ci.yml
  .gitignore
  README.md
  report.md
  requirements.txt
```

## Dataset policy

The real Kaggle dataset is **not included in this repository** because it is large and should be downloaded separately. Generated output files are also not committed because they can be reproduced by running the pipeline.

Ignored local paths include:

```text
data/*
outputs/*
__pycache__/
*.pyc
.venv/
```

The repository keeps only placeholder files:

```text
data/.gitkeep
outputs/.gitkeep
```

## Download the Kaggle dataset separately

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026
```

After downloading, place the per-asset CSV files inside:

```text
data/anonymized_data/
```

Expected local structure after download:

```text
data/
  anonymized_data/
    Asset_001.csv
    Asset_002.csv
    Asset_003.csv
    ...
    Asset_100.csv
```

Do **not** commit this folder. It is intentionally ignored by `.gitignore`.

## Expected data format

The main pipeline expects one combined long-format CSV:

```text
date,ticker,open,high,low,close,volume
```

If the Kaggle dataset is provided as one CSV per asset, use the combiner script to create `data/daily_prices.csv`.

Common aliases such as `symbol` for ticker, `datetime` for date, `adj_close` for close, and `vol` for volume are handled where possible.

## Windows setup without virtual environment

From the repository root:

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
python -m pip install --upgrade --user pip
python -m pip install --user -r requirements.txt
```

If `python` is not found, install Python first:

```powershell
winget install -e --id Python.Python.3.11
```

Then close and reopen PowerShell.

## Optional virtual environment setup

A virtual environment is optional. If you prefer one, run:

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run a smoke test with synthetic data

This verifies that the full pipeline works without needing the Kaggle dataset:

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
python scripts\run_pipeline.py --generate-synthetic
```

This creates a local synthetic file:

```text
data/sample_daily_prices.csv
```

## Combine the Kaggle per-asset CSV files

After placing Kaggle files in `data/anonymized_data/`, combine them:

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
python scripts\combine_asset_csvs.py --input data\anonymized_data --output data\daily_prices.csv
```

Expected output:

```text
data/daily_prices.csv
```

Do **not** commit `data/daily_prices.csv`.

## Run the final strategy on real data

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
python scripts\run_pipeline.py --data data\daily_prices.csv --config config\long_only_config.json
```

If `config/long_only_config.json` is not present, use the default config:

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
python scripts\run_pipeline.py --data data\daily_prices.csv
```

## Outputs generated locally

After a successful run, outputs are generated under:

```text
outputs/
```

Important files:

```text
outputs/cleaning_report.json
outputs/feature_manifest.csv
outputs/feature_frame.csv
outputs/predictions.csv
outputs/weights.csv
outputs/backtest_daily.csv
outputs/metrics_costs.json
outputs/metrics_no_costs.json
outputs/benchmark_metrics.json
outputs/pairs.csv
outputs/plots/equity_curve.png
outputs/plots/drawdown_curve.png
outputs/plots/top_pair_spread.png
outputs/models/latest_model.joblib
outputs/logs/pipeline.log
```

These files are ignored by Git because they are generated artifacts and may be large.

## Tests

Run tests from the repository root:

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
python -m pytest -q
```

## Git hygiene

Before committing, check that datasets and outputs are not staged:

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
git status
git ls-files data
git ls-files outputs
```

Expected tracked files should be only:

```text
data/.gitkeep
outputs/.gitkeep
```

If dataset/output files were accidentally staged, remove them from Git tracking while keeping local copies:

```powershell
cd "C:\Users\dell\Downloads\JPMC\Algorithmic-Equity-Trading-Pipeline-with-Statistical-Arbitrage-Overlay"
git rm -r --cached --ignore-unmatch data
git rm -r --cached --ignore-unmatch outputs
git add data\.gitkeep outputs\.gitkeep .gitignore README.md
```

## Compute note

This implementation uses pandas, NumPy, scikit-learn Ridge regression, matplotlib, and statsmodels. It is CPU-based. A local CPU is enough for the default pipeline. A GPU is not required unless the modelling layer is replaced with deep learning or GPU-specific gradient boosting.

## Methodology summary

### Cleaning

The pipeline normalizes columns, parses dates, sorts by ticker/date, removes duplicate ticker-date rows, removes invalid OHLC rows, forward-fills short gaps using only past data, fills missing volume with ticker-level median volume, and flags extreme return/volume anomalies for diagnostics.

### Features

Features include returns, volatility, momentum, trend ratios, RSI, ATR, open-to-close behavior, overnight gap, dollar volume, volume z-score, rolling beta, and cross-sectional ranks.

### Model

The model predicts 5-day forward returns using chronological walk-forward training to avoid lookahead bias. It combines Ridge regression with rule-based alpha features.

### Strategy

The final strategy ranks assets cross-sectionally and constructs a long-only top-ranked portfolio with weekly rebalancing. Backtests are reported both with and without transaction costs.

### Statistical arbitrage overlay

The stat-arb module searches for co-moving assets using return correlation, Engle-Granger cointegration tests, OLS hedge ratios, spread z-scores, and mean-reversion half-life. Candidate pairs can be used as a small relative-value sleeve alongside the main alpha portfolio.
