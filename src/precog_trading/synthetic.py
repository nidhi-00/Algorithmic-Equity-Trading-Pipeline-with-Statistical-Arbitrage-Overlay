
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_prices(
    out_path: str | Path,
    n_tickers: int = 8,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    dates = pd.bdate_range(start, end)
    tickers = [f"STK{i:02d}" for i in range(1, n_tickers + 1)]
    market = rng.normal(0.0003, 0.01, len(dates))
    sector_a = rng.normal(0.0001, 0.007, len(dates))
    sector_b = rng.normal(0.0001, 0.006, len(dates))
    rows = []
    base_prices = rng.uniform(20, 120, n_tickers)

    stk01_close = None
    for i, ticker in enumerate(tickers):
        px = np.empty(len(dates))
        px[0] = base_prices[i]
        beta_mkt = rng.uniform(0.6, 1.4)
        beta_sec = rng.uniform(0.5, 1.2)
        noise = rng.normal(0, 0.012, len(dates))
        sector = sector_a if i < n_tickers / 2 else sector_b
        for t in range(1, len(dates)):
            ret = beta_mkt * market[t] + beta_sec * sector[t] + noise[t]
            px[t] = max(1.0, px[t - 1] * np.exp(ret))
        if i == 0:
            stk01_close = px.copy()
        if i == 1 and stk01_close is not None:
            spread = np.zeros(len(dates))
            for t in range(1, len(dates)):
                spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.01)
            px = np.exp(np.log(stk01_close) + 0.3 * spread)
        close = px
        open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.02, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.02, len(dates)))
        volume = rng.lognormal(mean=13.0, sigma=0.25, size=len(dates))
        for d, o, h, l, c, v in zip(dates, open_, high, low, close, volume):
            rows.append({"date": d, "ticker": ticker, "open": round(float(o), 4), "high": round(float(h), 4), "low": round(float(l), 4), "close": round(float(c), 4), "volume": float(v)})

    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    missing_idx = rng.choice(df.index, size=max(10, len(df) // 200), replace=False)
    df.loc[missing_idx, "open"] = np.nan
    vol_missing_idx = rng.choice(df.index, size=max(10, len(df) // 250), replace=False)
    df.loc[vol_missing_idx, "volume"] = np.nan
    outlier_idx = rng.choice(df.index, size=max(5, len(df) // 700), replace=False)
    df.loc[outlier_idx, "close"] *= 1.35
    df.loc[outlier_idx, "high"] = df.loc[outlier_idx, ["open", "close"]].max(axis=1) * 1.02
    df.loc[outlier_idx, "low"] = df.loc[outlier_idx, ["open", "close"]].min(axis=1) * 0.98
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
