
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _cross_sectional_rank(df: pd.DataFrame, col: str, date_col: str = "date") -> pd.Series:
    return df.groupby(date_col)[col].rank(pct=True)


def engineer_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    out_parts = []

    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        g["ret_1"] = g["close"].pct_change()
        g["ret_5"] = g["close"].pct_change(5)
        g["ret_10"] = g["close"].pct_change(10)
        g["ret_20"] = g["close"].pct_change(20)

        log_ret_1 = np.log(g["close"]).diff()
        g["vol_5"] = log_ret_1.rolling(5, min_periods=3).std()
        g["vol_20"] = log_ret_1.rolling(20, min_periods=10).std()

        g["intraday_ret"] = (g["close"] - g["open"]) / g["open"]
        prev_close = g["close"].shift(1)
        g["overnight_gap"] = (g["open"] - prev_close) / prev_close

        g["sma_5"] = g["close"].rolling(5, min_periods=5).mean()
        g["sma_20"] = g["close"].rolling(20, min_periods=20).mean()
        g["price_to_sma_20"] = g["close"] / g["sma_20"] - 1.0
        g["sma_5_20_spread"] = g["sma_5"] / g["sma_20"] - 1.0
        g["rsi_14"] = _rsi(g["close"], 14) / 100.0

        prev_close = g["close"].shift(1)
        tr = pd.concat(
            [
                (g["high"] - g["low"]).abs(),
                (g["high"] - prev_close).abs(),
                (g["low"] - prev_close).abs(),
            ], axis=1,
        ).max(axis=1)
        g["atr_14"] = tr.rolling(14, min_periods=7).mean() / g["close"]

        g["dollar_vol"] = g["close"] * g["volume"]
        g["log_dollar_vol"] = np.log1p(g["dollar_vol"])
        log_volume = np.log1p(g["volume"])
        vol_mean_20 = log_volume.rolling(20, min_periods=5).mean()
        vol_std_20 = log_volume.rolling(20, min_periods=5).std()
        g["volume_z_20"] = (log_volume - vol_mean_20) / vol_std_20.replace(0, np.nan)

        g["target_fwd_return"] = g["close"].shift(-horizon) / g["close"] - 1.0
        out_parts.append(g)

    df = pd.concat(out_parts, ignore_index=True)
    market_by_date = df.groupby("date")["ret_1"].mean().rename("market_ret_1")
    df = df.merge(market_by_date, on="date", how="left")

    out_parts = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        rolling_cov = (g["ret_1"] * g["market_ret_1"]).rolling(20, min_periods=10).mean() - (
            g["ret_1"].rolling(20, min_periods=10).mean()
            * g["market_ret_1"].rolling(20, min_periods=10).mean()
        )
        rolling_var = g["market_ret_1"].rolling(20, min_periods=10).var()
        g["beta_20"] = rolling_cov / rolling_var.replace(0, np.nan)
        out_parts.append(g)

    df = pd.concat(out_parts, ignore_index=True)
    df["cs_momentum_rank"] = _cross_sectional_rank(df, "ret_20")
    df["cs_vol_rank"] = _cross_sectional_rank(df, "vol_20")

    winsor_cols = [
        "ret_1", "ret_5", "ret_10", "ret_20", "vol_5", "vol_20", "intraday_ret",
        "overnight_gap", "price_to_sma_20", "sma_5_20_spread", "volume_z_20", "beta_20",
    ]
    for col in winsor_cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def build_model_frame(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    features = list(features)
    keep = features + ["date", "ticker", "target_fwd_return", "close"]
    out = df[keep].copy()
    out = out.dropna(subset=features + ["target_fwd_return"])
    return out.reset_index(drop=True)
