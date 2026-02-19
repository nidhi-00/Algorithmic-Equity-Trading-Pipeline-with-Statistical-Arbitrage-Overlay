from __future__ import annotations

import pandas as pd


def predictions_to_weights(
    pred_df: pd.DataFrame,
    top_quantile: float = 0.1,
    bottom_quantile: float = 0.0,
    rebalance_frequency: str = "W-FRI",
) -> pd.DataFrame:
    """Convert cross-sectional predictions into long-only target weights.

    Final strategy:
    - Rank all stocks by model prediction each day.
    - Select the top prediction bucket.
    - Allocate capital only to the selected long book.
    - Rebalance weekly to reduce transaction costs.
    - Avoid short exposure because the 2024-2026 test period has strong positive market drift.
    """
    del bottom_quantile

    out = pred_df.copy()
    out["date"] = pd.to_datetime(out["date"])

    def _build_for_day(day: pd.DataFrame) -> pd.DataFrame:
        scores = day["prediction"].replace([float("inf"), float("-inf")], pd.NA)

        top_cut = scores.quantile(1 - top_quantile)
        long_mask = scores >= top_cut

        day = day.copy()
        day["weight"] = 0.0

        selected = day.loc[long_mask].copy()
        if selected.empty:
            return day

        positive_scores = selected["prediction"].clip(lower=0.0)

        if positive_scores.sum() > 0:
            weights = positive_scores / positive_scores.sum()
        else:
            weights = pd.Series(1.0 / len(selected), index=selected.index)

        day.loc[selected.index, "weight"] = weights
        return day

    out = out.groupby("date", group_keys=False).apply(_build_for_day).reset_index(drop=True)

    if rebalance_frequency != "D":
        pivot = out.pivot(index="date", columns="ticker", values="weight").sort_index()

        rebalance_dates = []
        for _, group in pivot.groupby(pd.Grouper(freq=rebalance_frequency)):
            if not group.empty:
                rebalance_dates.append(group.index.max())

        mask = pivot.index.isin(rebalance_dates)
        pivot.loc[~mask, :] = pd.NA
        pivot = pivot.ffill().fillna(0.0)

        melted = pivot.stack().rename("weight").reset_index()
        preds = out[["date", "ticker", "prediction"]]
        out = melted.merge(preds, on=["date", "ticker"], how="left")

    return out[["date", "ticker", "weight", "prediction"]]
