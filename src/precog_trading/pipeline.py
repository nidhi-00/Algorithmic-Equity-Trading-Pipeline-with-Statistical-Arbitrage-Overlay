
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .backtest import backtest_long_short, make_equal_weight_benchmark, save_backtest_plots
from .cleaning import clean_prices
from .features import build_model_frame, engineer_features
from .io import ensure_required_columns, load_prices_csv
from .modelling import walk_forward_predictions
from .signals import predictions_to_weights
from .statarb import discover_pairs, make_pair_spread_plot
from .synthetic import generate_synthetic_prices


def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline.log"
    logger = logging.getLogger("precog_trading")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    data_path: str | Path | None,
    project_root: str | Path,
    config_path: str | Path | None = None,
    generate_synthetic: bool = False,
) -> dict[str, object]:
    project_root = Path(project_root)
    config_path = Path(config_path or project_root / "config" / "default_config.json")
    config = _load_config(config_path)
    (project_root / "outputs").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs" / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(project_root / "outputs" / "logs")
    logger.info("Starting pipeline")

    if generate_synthetic or data_path is None:
        data_path = project_root / "data" / "sample_daily_prices.csv"
        logger.info("Generating synthetic dataset at %s", data_path)
        generate_synthetic_prices(data_path, random_state=config["random_state"])

    df = load_prices_csv(data_path)
    ensure_required_columns(df, config["required_columns"])
    logger.info("Loaded %s rows across %s tickers", len(df), df["ticker"].nunique())

    cleaned = clean_prices(df, max_forward_fill_days=config["max_forward_fill_days"])
    clean_df = cleaned.data
    with open(project_root / "outputs" / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(cleaned.report, f, indent=2)
    logger.info("Cleaning complete: %s rows remain", len(clean_df))

    feat_df = engineer_features(clean_df, horizon=config["prediction_horizon_days"])
    feat_df.to_csv(project_root / "outputs" / "feature_frame.csv", index=False)
    feature_manifest = pd.DataFrame({
        "feature": config["feature_columns"],
        "description": [
            "1-day return", "5-day return", "10-day return", "20-day return", "5-day volatility",
            "20-day volatility", "14-day RSI scaled to [0,1]", "14-day ATR divided by price",
            "Price relative to 20-day moving average", "5-day versus 20-day moving-average spread",
            "Open-to-close intraday return", "Open versus previous close gap", "Log dollar volume",
            "20-day rolling log-volume z-score", "20-day beta to equal-weight universe return",
            "Daily cross-sectional rank of 20-day momentum", "Daily cross-sectional rank of 20-day volatility",
        ],
    })
    feature_manifest.to_csv(project_root / "outputs" / "feature_manifest.csv", index=False)

    model_df = build_model_frame(feat_df, config["feature_columns"])
    all_dates = sorted(model_df["date"].unique())
    test_start_idx = max(0, len(all_dates) - (252 * config["test_years"]))
    test_start_date = all_dates[test_start_idx]
    logger.info("Test set starts on %s", test_start_date)

    pred_df, _ = walk_forward_predictions(
        model_df=model_df,
        features=config["feature_columns"],
        train_window_days=config["train_window_days"],
        test_start_date=test_start_date,
        random_state=config["random_state"],
        ridge_alpha=config["ridge_alpha"],
        model_output_path=project_root / "outputs" / "models" / "latest_model.joblib",
    )
    pred_df.to_csv(project_root / "outputs" / "predictions.csv", index=False)
    weights_df = predictions_to_weights(
        pred_df,
        top_quantile=config["top_quantile"],
        bottom_quantile=config["bottom_quantile"],
        rebalance_frequency=config["rebalance_frequency"],
    )
    weights_df.to_csv(project_root / "outputs" / "weights.csv", index=False)

    feat_test = feat_df.loc[feat_df["date"] >= test_start_date].copy()
    backtest_costs, metrics_costs, benchmark_metrics = backtest_long_short(
        feature_df=feat_test,
        weights_df=weights_df,
        initial_capital=config["initial_capital"],
        transaction_cost_bps=config["transaction_cost_bps"],
    )
    backtest_costs.to_csv(project_root / "outputs" / "backtest_daily.csv", index=False)
    with open(project_root / "outputs" / "metrics_costs.json", "w", encoding="utf-8") as f:
        json.dump(metrics_costs, f, indent=2)
    with open(project_root / "outputs" / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_metrics, f, indent=2)

    _, metrics_nocosts, _ = backtest_long_short(
        feature_df=feat_test,
        weights_df=weights_df,
        initial_capital=config["initial_capital"],
        transaction_cost_bps=0.0,
    )
    with open(project_root / "outputs" / "metrics_no_costs.json", "w", encoding="utf-8") as f:
        json.dump(metrics_nocosts, f, indent=2)

    benchmark_df = make_equal_weight_benchmark(feat_test, weights_df, config["initial_capital"])
    save_backtest_plots(backtest_costs, benchmark_df, project_root / "outputs" / "plots")

    pairs = discover_pairs(
        clean_df[["date", "ticker", "close"]],
        lookback_days=config["pair_lookback_days"],
        min_correlation=config["pair_min_correlation"],
        max_pvalue=config["pair_max_pvalue"],
        top_k=config["pair_top_k"],
    )
    pairs.to_csv(project_root / "outputs" / "pairs.csv", index=False)
    pair_plot = None
    if not pairs.empty:
        pair_plot = make_pair_spread_plot(
            clean_df[["date", "ticker", "close"]],
            pairs.iloc[0],
            project_root / "outputs" / "plots" / "top_pair_spread.png",
        )

    summary = {
        "data_path": str(data_path),
        "test_start_date": str(test_start_date),
        "metrics_costs": metrics_costs,
        "metrics_no_costs": metrics_nocosts,
        "benchmark_metrics": benchmark_metrics,
        "pairs_found": int(len(pairs)),
        "top_pair_plot": pair_plot,
    }
    with open(project_root / "outputs" / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Pipeline complete")
    return summary
