
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import summarise_backtest


def _wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="ticker", values=value_col).sort_index().fillna(0.0)


def make_equal_weight_benchmark(feature_df: pd.DataFrame, weights_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    active_tickers = weights_df.loc[weights_df["weight"] != 0, "ticker"].unique().tolist()
    returns_w = feature_df.pivot(index="date", columns="ticker", values="ret_1").sort_index()
    if active_tickers:
        returns_w = returns_w[active_tickers]
    returns_w = returns_w.fillna(0.0)
    bench_ret = returns_w.mean(axis=1)
    bench_equity = initial_capital * (1.0 + bench_ret).cumprod()
    return pd.DataFrame({"date": bench_ret.index, "net_return": bench_ret.values, "turnover": 0.0, "equity": bench_equity.values})


def backtest_long_short(
    feature_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    initial_capital: float,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    returns = feature_df[["date", "ticker", "ret_1"]].copy().dropna()
    returns_w = _wide(returns, "ret_1")

    target_w = _wide(weights_df, "weight").reindex(returns_w.index).fillna(0.0)
    held_w = target_w.shift(1).fillna(0.0)

    turnover = target_w.diff().abs().sum(axis=1)
    turnover.iloc[0] = target_w.iloc[0].abs().sum()
    gross = held_w.mul(returns_w).sum(axis=1)
    costs = (transaction_cost_bps / 10000.0) * turnover
    net = gross - costs
    equity = initial_capital * (1.0 + net).cumprod()

    out = pd.DataFrame(
        {"gross_return": gross, "transaction_cost": costs, "net_return": net, "turnover": turnover, "equity": equity},
        index=returns_w.index,
    ).reset_index().rename(columns={"index": "date"})

    metrics = summarise_backtest(out)
    benchmark_df = make_equal_weight_benchmark(feature_df, weights_df, initial_capital)
    benchmark_metrics = summarise_backtest(benchmark_df)
    return out, metrics, benchmark_metrics


def save_backtest_plots(backtest_df: pd.DataFrame, benchmark_df: pd.DataFrame, out_dir: str | Path) -> tuple[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_path = str(out_dir / "equity_curve.png")
    dd_path = str(out_dir / "drawdown_curve.png")

    plt.figure(figsize=(10, 5))
    plt.plot(backtest_df["date"], backtest_df["equity"], label="Strategy")
    plt.plot(benchmark_df["date"], benchmark_df["equity"], label="Equal-weight benchmark")
    plt.title("Cumulative equity curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(equity_path, dpi=150)
    plt.close()

    strategy_dd = backtest_df["equity"] / backtest_df["equity"].cummax() - 1.0
    benchmark_dd = benchmark_df["equity"] / benchmark_df["equity"].cummax() - 1.0
    plt.figure(figsize=(10, 5))
    plt.plot(backtest_df["date"], strategy_dd, label="Strategy drawdown")
    plt.plot(benchmark_df["date"], benchmark_dd, label="Benchmark drawdown")
    plt.title("Drawdown comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dd_path, dpi=150)
    plt.close()
    return equity_path, dd_path
