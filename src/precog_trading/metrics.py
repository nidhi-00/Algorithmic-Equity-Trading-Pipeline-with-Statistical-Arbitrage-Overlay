
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def sharpe_ratio(returns: pd.Series, annualisation: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    vol = returns.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(np.sqrt(annualisation) * returns.mean() / vol)


def summarise_backtest(df: pd.DataFrame) -> dict[str, float]:
    dd = compute_drawdown(df["equity"])
    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0
    n_days = max(len(df), 1)
    ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else float("nan")
    avg_daily_turnover = float(df["turnover"].mean())
    return {
        "total_return": float(total_return),
        "annualised_return": float(ann_return),
        "annualised_sharpe": sharpe_ratio(df["net_return"]),
        "max_drawdown": float(dd.min()),
        "average_drawdown": float(dd.mean()),
        "average_daily_turnover": avg_daily_turnover,
        "annualised_turnover": float(avg_daily_turnover * 252),
        "final_equity": float(df["equity"].iloc[-1]),
    }
