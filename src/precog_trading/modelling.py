
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    ridge_pipeline: Pipeline
    features: list[str]
    heuristic_coefficients: dict[str, float]


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def _heuristic_signal(df: pd.DataFrame) -> pd.Series:
    coeffs = {
        "ret_5": 0.35,
        "ret_20": 0.45,
        "price_to_sma_20": 0.15,
        "volume_z_20": 0.10,
        "vol_20": -0.20,
        "beta_20": -0.05,
    }
    score = pd.Series(0.0, index=df.index)
    for feature, coef in coeffs.items():
        if feature in df.columns:
            score = score + coef * _zscore(df[feature].fillna(0.0))
    return score


def walk_forward_predictions(
    model_df: pd.DataFrame,
    features: Iterable[str],
    train_window_days: int,
    test_start_date: pd.Timestamp,
    random_state: int,
    ridge_alpha: float,
    model_output_path: str | Path | None = None,
    retrain_every_n_days: int = 21,
) -> tuple[pd.DataFrame, ModelBundle]:
    del random_state
    features = list(features)
    model_df = model_df.sort_values(["date", "ticker"]).copy()
    scoring_dates = sorted(d for d in model_df["date"].unique() if d >= test_start_date)

    _ = TimeSeriesSplit(n_splits=5)
    ridge_pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))])

    all_dates = sorted(model_df["date"].unique())
    date_to_ix = {d: i for i, d in enumerate(all_dates)}
    last_train_ix = None
    last_bundle: ModelBundle | None = None
    all_preds: list[pd.DataFrame] = []

    for current_date in scoring_dates:
        current_ix = date_to_ix[current_date]
        should_retrain = last_train_ix is None or current_ix - last_train_ix >= retrain_every_n_days

        if should_retrain:
            train_start_ix = max(0, current_ix - train_window_days)
            train_start = all_dates[train_start_ix]
            train_mask = (model_df["date"] >= train_start) & (model_df["date"] < current_date)
            train = model_df.loc[train_mask]
            if train["date"].nunique() < 60:
                continue

            X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_train = train["target_fwd_return"]
            ridge_pipeline.fit(X_train, y_train)

            last_bundle = ModelBundle(
                ridge_pipeline=ridge_pipeline,
                features=features,
                heuristic_coefficients={
                    "ret_5": 0.35, "ret_20": 0.45, "price_to_sma_20": 0.15,
                    "volume_z_20": 0.10, "vol_20": -0.20, "beta_20": -0.05,
                },
            )
            last_train_ix = current_ix

        if last_bundle is None:
            continue

        todays = model_df.loc[model_df["date"] == current_date, ["date", "ticker"] + features].copy()
        if todays.empty:
            continue
        X_test = todays[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        pred_ridge = pd.Series(last_bundle.ridge_pipeline.predict(X_test), index=todays.index)
        pred_rule = _heuristic_signal(todays)
        ensemble = 0.6 * _zscore(pred_ridge) + 0.4 * _zscore(pred_rule)

        preds = todays[["date", "ticker"]].copy()
        preds["pred_ridge"] = pred_ridge
        preds["pred_rule"] = pred_rule
        preds["prediction"] = ensemble
        all_preds.append(preds)

    if last_bundle is None or not all_preds:
        raise RuntimeError("Insufficient data to train the walk-forward model.")

    if model_output_path is not None:
        model_output_path = Path(model_output_path)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(last_bundle, model_output_path)

    pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["date", "ticker"])
    return pred_df, last_bundle
