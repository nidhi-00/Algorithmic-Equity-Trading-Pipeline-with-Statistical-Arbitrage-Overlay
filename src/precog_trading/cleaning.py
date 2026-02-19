
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CleaningResult:
    data: pd.DataFrame
    report: dict[str, Any]


def _bounded_ffill(group: pd.DataFrame, columns: list[str], limit: int) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    group[columns] = group[columns].ffill(limit=limit)
    return group


def clean_prices(df: pd.DataFrame, max_forward_fill_days: int = 3) -> CleaningResult:
    """Clean OHLCV data conservatively without introducing future information."""
    report: dict[str, Any] = {
        "initial_rows": int(len(df)),
        "duplicate_rows_removed": 0,
        "rows_removed_invalid_prices": 0,
        "rows_remaining_after_cleaning": 0,
        "missing_before": df.isna().sum().to_dict(),
        "missing_after": {},
        "anomaly_counts": {},
        "ticker_count": int(df["ticker"].nunique()),
        "date_min": str(pd.to_datetime(df["date"]).min()),
        "date_max": str(pd.to_datetime(df["date"]).max()),
    }

    df = df.sort_values(["ticker", "date"]).copy()

    dup_count = int(df.duplicated(subset=["date", "ticker"]).sum())
    report["duplicate_rows_removed"] = dup_count
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    numeric_cols = [col for col in ["open", "high", "low", "close", "volume"] if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_price_mask = (
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    report["rows_removed_invalid_prices"] = int(invalid_price_mask.sum())
    df = df.loc[~invalid_price_mask].copy()

    filled_parts = []
    for _, group in df.groupby("ticker", sort=False):
        filled_parts.append(_bounded_ffill(group, ["open", "high", "low", "close"], max_forward_fill_days))
    df = pd.concat(filled_parts, ignore_index=True)

    median_vol = df.groupby("ticker")["volume"].transform("median")
    df["volume"] = df["volume"].fillna(median_vol).fillna(0.0).clip(lower=0.0)

    px_max = df[["open", "close", "high"]].max(axis=1)
    px_min = df[["open", "close", "low"]].min(axis=1)
    df["high"] = px_max
    df["low"] = px_min

    required_price_cols = ["open", "high", "low", "close"]
    still_missing = df[required_price_cols].isna().any(axis=1)
    df = df.loc[~still_missing].copy()

    df["ret_1_raw"] = df.groupby("ticker")["close"].pct_change()
    vol_med = df.groupby("ticker")["volume"].transform(lambda s: s.rolling(20, min_periods=5).median())
    df["is_return_anomaly"] = df["ret_1_raw"].abs() > 0.30
    df["is_volume_anomaly"] = df["volume"] > (8.0 * vol_med.fillna(np.inf))

    report["anomaly_counts"] = {
        "return_anomaly_rows": int(df["is_return_anomaly"].sum()),
        "volume_anomaly_rows": int(df["is_volume_anomaly"].sum()),
    }
    report["rows_remaining_after_cleaning"] = int(len(df))
    report["missing_after"] = df.isna().sum().to_dict()
    return CleaningResult(data=df.reset_index(drop=True), report=report)
