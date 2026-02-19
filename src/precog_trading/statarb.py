
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def estimate_half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 20:
        return float("nan")
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    aligned = pd.concat([delta, lagged], axis=1).dropna()
    aligned.columns = ["delta", "lagged"]
    x = sm.add_constant(aligned["lagged"])
    model = sm.OLS(aligned["delta"], x).fit()
    beta = model.params["lagged"]
    if beta >= 0:
        return float("nan")
    return float(-np.log(2.0) / beta)


def discover_pairs(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    min_correlation: float = 0.55,
    max_pvalue: float = 0.05,
    top_k: int = 10,
) -> pd.DataFrame:
    pivot_close = prices.pivot(index="date", columns="ticker", values="close").sort_index().tail(lookback_days)
    log_px = np.log(pivot_close)
    rets = pivot_close.pct_change().dropna(how="all")
    results = []
    for a, b in combinations(list(pivot_close.columns), 2):
        pair_px = log_px[[a, b]].dropna()
        pair_ret = rets[[a, b]].dropna()
        if len(pair_px) < max(100, lookback_days // 2):
            continue
        corr = pair_ret[a].corr(pair_ret[b])
        if pd.isna(corr) or corr < min_correlation:
            continue
        pvalue = coint(pair_px[a], pair_px[b])[1]
        x = sm.add_constant(pair_px[b])
        hedge = sm.OLS(pair_px[a], x).fit()
        beta = hedge.params[b]
        spread = pair_px[a] - beta * pair_px[b]
        half_life = estimate_half_life(spread)
        if pd.isna(half_life) or pvalue > max_pvalue:
            continue
        score = corr * (1.0 - pvalue) / (1.0 + abs(half_life - 20) / 20)
        results.append({
            "ticker_y": a,
            "ticker_x": b,
            "corr": float(corr),
            "cointegration_pvalue": float(pvalue),
            "hedge_ratio": float(beta),
            "half_life_days": float(half_life),
            "score": float(score),
        })
    res = pd.DataFrame(results)
    if res.empty:
        return res
    return res.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


def make_pair_spread_plot(prices: pd.DataFrame, pair_row: pd.Series, out_path: str | Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = pair_row["ticker_y"]
    b = pair_row["ticker_x"]
    beta = pair_row["hedge_ratio"]
    pivot = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    log_px = np.log(pivot[[a, b]].dropna())
    spread = log_px[a] - beta * log_px[b]
    z = (spread - spread.rolling(60, min_periods=20).mean()) / spread.rolling(60, min_periods=20).std()

    plt.figure(figsize=(10, 5))
    plt.plot(z.index, z, label=f"{a} - {beta:.2f} * {b}")
    plt.axhline(0.0, linestyle="--")
    plt.axhline(2.0, linestyle=":")
    plt.axhline(-2.0, linestyle=":")
    plt.title("Pair spread z-score")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)
