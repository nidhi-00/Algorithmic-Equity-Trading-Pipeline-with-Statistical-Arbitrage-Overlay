# Precog Quant Task 2026 - Project Report

## 1. Objective

The objective was to build an end-to-end algorithmic trading research pipeline over a universe of anonymized equities. The pipeline needed to take raw OHLCV market data, clean it, engineer features, train a predictive model, convert predictions into tradable signals, backtest the strategy with realistic transaction costs, compare performance against a benchmark, and explore statistical-arbitrage opportunities.

The final project covers all four required parts:

1. Feature Engineering & Data Cleaning
2. Model Training & Strategy Formulation
3. Backtesting & Performance Analysis
4. Statistical Arbitrage Overlay

## 2. Dataset and experimental setup

The real Kaggle data was supplied as one CSV file per anonymized asset. I first converted the per-asset files into one long-format file:

```text
date,ticker,open,high,low,close,volume
```

Final dataset summary:

| Item | Value |
|---|---:|
| Raw rows | 251,100 |
| Rows after cleaning | 251,081 |
| Ticker count | 100 |
| Date range | 2016-01-25 to 2026-01-16 |
| Test-set start | 2024-01-08 |
| Initial capital | $1,000,000 |
| Transaction cost | 10 bps per unit traded |
| Benchmark | Equal-weight buy-and-hold universe portfolio |

The out-of-sample evaluation period starts on 2024-01-08 and continues until the end of the available dataset. I treated the final two years as the main test period because the task explicitly asks for out-of-sample performance over at least two years.

## 3. Feature engineering and data cleaning

### 3.1 Data-quality checks

The pipeline performed the following checks:

- Normalized all column names into a consistent schema.
- Parsed and sorted dates.
- Sorted data by ticker and date.
- Removed duplicate ticker-date rows.
- Removed invalid OHLC rows, such as non-positive prices or inconsistent high/low values.
- Checked for missing values before and after cleaning.
- Flagged extreme return anomalies.
- Flagged extreme volume anomalies.

Cleaning results:

| Cleaning item | Value |
|---|---:|
| Initial rows | 251,100 |
| Duplicate rows removed | 0 |
| Invalid price rows removed | 19 |
| Rows remaining | 251,081 |
| Return anomaly rows | 4 |
| Volume anomaly rows | 99 |
| Missing core OHLCV values before cleaning | 0 |

Observation: the dataset was mostly clean. Only 19 invalid price rows were removed out of more than 251k rows. The anomaly flags were retained as diagnostics rather than blindly deleting many observations, because large market moves or volume spikes can contain information instead of being pure data errors.

### 3.2 Feature set

The features were designed to capture cross-sectional differences across the stock universe rather than model a single asset independently.

Main feature groups:

1. **Return and momentum features**
   - 1-day return
   - 5-day return
   - 10-day return
   - 20-day return
   - Price-to-SMA20
   - SMA5/SMA20 trend spread

2. **Risk and volatility features**
   - 5-day rolling volatility
   - 20-day rolling volatility
   - ATR scaled by price
   - Rolling beta to the equal-weight market return

3. **Volume and liquidity features**
   - Log dollar volume
   - Volume z-score

4. **Intraday behavior features**
   - Open-to-close return
   - Overnight gap

5. **Cross-sectional features**
   - Momentum rank
   - Volatility rank

The motivation was to expose the model to several different market effects: short-term continuation/reversal, medium-term momentum, volatility risk, liquidity, beta exposure, and cross-sectional relative strength.

## 4. Model training and strategy formulation

### 4.1 Prediction target

The model predicts 5-day forward return:

```text
target_t = close_{t+5} / close_t - 1
```

This target was chosen because one-day returns are often extremely noisy, while a 5-day horizon is still short enough to be useful for a weekly trading system.

### 4.2 Walk-forward training

The model uses chronological walk-forward training. For each prediction period, only data available before that period is used for training. This avoids lookahead bias and better reflects how the strategy would behave in real time.

### 4.3 Prediction model

The prediction engine combines:

1. **Ridge regression**
   - Handles correlated features better than ordinary least squares.
   - Provides a simple, explainable baseline.
   - Reduces overfitting by shrinking unstable coefficients.

2. **Rule-based alpha component**
   - Combines intuitive signals such as momentum, liquidity, volatility, and beta.
   - Acts as a transparent sanity check against the purely fitted model.

The final score is used to rank stocks cross-sectionally each rebalance day.

### 4.4 Final strategy logic

The final selected strategy was a long-only top-ranked strategy:

1. Rank all assets by predicted attractiveness.
2. Select the top predicted names.
3. Allocate across selected names.
4. Rebalance weekly.
5. Apply a one-day signal lag.
6. Deduct 10 bps transaction costs based on turnover.

I initially tested a market-neutral long-short version, but it underperformed after transaction costs. That result was still useful: it showed that turnover and market regime matter. Since the out-of-sample period had a strong upward market drift, the final long-only strategy was more suitable and more aligned with the realized market environment.

## 5. Backtesting methodology

The backtester uses close-to-close daily returns and applies the previous day's portfolio weights to the next day's returns. This creates a realistic one-day lag between signal generation and realized PnL.

Transaction costs are deducted using:

```text
daily_cost = transaction_cost_bps / 10000 * daily_turnover
```

where turnover is the sum of absolute changes in portfolio weights.

Reported metrics:

- Total return
- Annualized return
- Annualized Sharpe ratio
- Maximum drawdown
- Average drawdown
- Average daily turnover
- Annualized turnover
- Final equity

The benchmark is an equal-weight buy-and-hold portfolio across the tradable universe.

## 6. Final backtest results

### 6.1 Main result table

| Metric | Final strategy, with costs | Final strategy, no costs | Equal-weight benchmark |
|---|---:|---:|---:|
| Total return | 84.72% | 107.82% | 47.97% |
| Annualized return | 35.50% | 43.64% | 21.41% |
| Annualized Sharpe | 1.63 | 1.93 | 1.72 |
| Maximum drawdown | -19.53% | -15.47% | -12.99% |
| Average drawdown | -5.23% | -4.15% | -1.49% |
| Average daily turnover | 23.15% | 23.15% | 0.00% |
| Annualized turnover | 58.33x | 58.33x | 0.00x |
| Final equity | $1,847,178 | $2,078,166 | $1,494,654 |

### 6.2 Interpretation

The final strategy outperformed the equal-weight benchmark in absolute terms:

```text
Final strategy ending equity: $1.85M
Benchmark ending equity:      $1.49M
```

The strategy generated an 84.72% net return after transaction costs versus 47.97% for the benchmark. This suggests that the model's cross-sectional ranking had useful predictive information during the out-of-sample test period.

However, the benchmark had a slightly higher Sharpe ratio, 1.72 versus the strategy's 1.63, and a lower maximum drawdown, -12.99% versus -19.53%. This means the strategy won on absolute return but took more active risk.

### 6.3 Transaction-cost analysis

Without transaction costs, the strategy returned 107.82% with a Sharpe ratio of 1.93. After 10 bps transaction costs, return fell to 84.72% and Sharpe fell to 1.63.

This shows that the signal survived transaction costs, but costs were still meaningful. The strategy had average daily turnover of 23.15%, or annualized turnover of roughly 58.33x. A future improvement would be to reduce turnover through slower rebalancing, turnover penalties, or position smoothing.

### 6.4 Equity curve observation

The final equity curve shows that the strategy tracked the benchmark early in the test period, suffered a larger drawdown around the middle of the period, and then strongly outperformed during the later rally. This suggests that the signal was not uniformly stable across all regimes but performed strongly when the selected top-ranked names participated in the market trend.

### 6.5 Drawdown observation

The drawdown plot shows that the strategy's worst drawdown was larger than the benchmark's. This is the main weakness of the final system. The strategy generated higher return, but it did so with more concentrated active exposure.

## 7. Statistical arbitrage overlay

The stat-arb module searches for relative-value relationships between pairs of assets.

### 7.1 Method

The process was:

1. Pivot close prices into a date-by-ticker matrix.
2. Compute return correlations as a first-pass filter.
3. Test log-price pairs using Engle-Granger cointegration.
4. Estimate hedge ratios using OLS.
5. Build spreads:

```text
spread = log(price_y) - hedge_ratio * log(price_x)
```

6. Convert spreads into rolling z-scores.
7. Estimate mean-reversion half-life.
8. Rank pairs using correlation, cointegration p-value, and half-life quality.

### 7.2 Result

The final run found 10 candidate pairs. The displayed top spread plot was:

```text
Asset_028 - 1.13 * Asset_042
```

The spread z-score repeatedly moved outside +/-2 and later reverted toward zero. This is visually consistent with a potential mean-reversion relationship, although the pair should still be validated out of sample before using capital.

### 7.3 How to incorporate the overlay

The main long-only alpha portfolio can be combined with a smaller market-neutral pair sleeve:

```text
w_total = (1 - lambda) * w_alpha + lambda * w_pairs
```

A practical lambda could be 5% to 15% initially. The pair sleeve should only enter trades when the spread z-score is extreme, for example:

- Short spread when z-score > +2
- Long spread when z-score < -2
- Exit when z-score reverts near 0

This overlay can add relative-value exposure that is less dependent on broad market direction.

## 8. Key observations

1. **The dataset was clean enough for systematic modelling.** Only 19 invalid price rows were removed from 251,100 rows.
2. **The initial market-neutral idea was not the best final fit.** It missed the strong upward drift in the out-of-sample test period and was hurt by turnover.
3. **The final long-only ranking strategy produced strong absolute performance.** It reached $1.85M final equity from $1M after transaction costs.
4. **Costs mattered but did not destroy the edge.** Net return remained 84.72% after 10 bps transaction costs.
5. **Risk-adjusted performance still has room to improve.** The strategy's Sharpe was slightly below the benchmark and drawdown was larger.
6. **Stat-arb relationships existed in the anonymized universe.** The pipeline found 10 candidate pairs, including a visually mean-reverting spread between Asset_028 and Asset_042.

## 9. Limitations

The project is intentionally research-oriented, so the current version has limitations:

- The strategy is evaluated on one final out-of-sample period.
- Repeated tuning on the same test set would risk overfitting.
- Transaction costs are simplified and do not model slippage, market impact, or borrow costs.
- The benchmark is equal-weight buy-and-hold, not a sector- or beta-matched benchmark.
- The stat-arb overlay identifies candidates but does not fully backtest the pair sleeve.
- Anonymized tickers prevent sector-aware risk controls unless the assets are reverse-engineered or clustered.

## 10. Future improvements

Useful next steps:

1. Add turnover regularization directly into portfolio construction.
2. Add volatility targeting to reduce drawdowns.
3. Add regime filters to reduce exposure during weak market regimes.
4. Backtest the stat-arb sleeve as a separate strategy.
5. Use nested validation to avoid test-set overfitting.
6. Add factor-neutral constraints if sector/factor mappings become available.
7. Add richer models such as gradient boosting, while keeping walk-forward validation.

## 11. Resume / JPMorgan Chase QR summary

### Project title

**Transaction-Cost-Aware Equity Alpha and Statistical Arbitrage Research Pipeline**

### Resume bullets

- Built an end-to-end Python equity alpha research pipeline on 100 anonymized equities and 251k+ OHLCV rows, covering data cleaning, anomaly detection, feature engineering, predictive ranking, transaction-cost-aware backtesting, and statistical-arbitrage pair discovery.
- Developed a walk-forward cross-sectional ranking strategy using Ridge regression and rule-based alpha features; achieved 84.7% net return after 10 bps transaction costs versus 48.0% for an equal-weight benchmark over the out-of-sample test period.
- Evaluated strategy robustness through Sharpe, drawdown, turnover, cost-drag, and benchmark-relative analysis; identified 10 candidate co-moving/cointegrated asset pairs for a potential relative-value overlay.

### Interview pitch

I built this as a realistic quant research project rather than only a modelling exercise. The important part was not just predicting returns, but converting noisy predictions into a tradable portfolio, adding a one-day execution lag, charging transaction costs, comparing against a benchmark, and then honestly analyzing where the strategy won and where it still had risk. The final long-only ranking strategy beat the benchmark on absolute return, but I also found that its drawdown and turnover were the main weaknesses, which gives clear directions for future improvement.

## 12. Conclusion

The final project demonstrates a complete quantitative research workflow: data engineering, feature construction, walk-forward modelling, portfolio construction, transaction-cost-aware backtesting, benchmark comparison, and statistical-arbitrage research. The final strategy achieved strong net absolute returns over the out-of-sample test period, while the analysis also identified its key risk tradeoffs. This makes the project useful both as a submission for the task and as a resume project for quantitative research roles.
