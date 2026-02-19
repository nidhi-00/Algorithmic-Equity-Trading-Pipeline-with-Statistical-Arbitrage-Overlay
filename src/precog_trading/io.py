
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_COLUMN_ALIASES = {
    "date": {"date", "datetime", "timestamp", "time"},
    "ticker": {"ticker", "symbol", "asset", "stock", "instrument"},
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "adj_close", "adjusted_close", "c"},
    "volume": {"volume", "vol", "v"},
}


def _normalise_column_name(name: str) -> str:
    cleaned = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    for canonical, aliases in _COLUMN_ALIASES.items():
        if cleaned in aliases:
            return canonical
    return cleaned


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected OHLCV column names."""
    renamed = {col: _normalise_column_name(col) for col in df.columns}
    df = df.rename(columns=renamed).copy()
    if "date" not in df.columns:
        raise ValueError("Input data must contain a date-like column.")
    if "ticker" not in df.columns:
        raise ValueError("Input data must contain a ticker-like column.")
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find CSV: {path}")
    df = pd.read_csv(path)
    return normalise_schema(df)


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
