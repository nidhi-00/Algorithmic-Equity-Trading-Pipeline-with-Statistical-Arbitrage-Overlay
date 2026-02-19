
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precog_trading.backtest import backtest_long_short


def test_backtest_static_weights_has_turnover_only_once():
    dates = pd.bdate_range("2024-01-01", periods=6)
    feat_rows = []
    weight_rows = []
    for d in dates:
        feat_rows.extend([{"date": d, "ticker": "A", "ret_1": 0.01}, {"date": d, "ticker": "B", "ret_1": -0.01}])
        weight_rows.extend([{"date": d, "ticker": "A", "weight": 0.5, "prediction": 1.0}, {"date": d, "ticker": "B", "weight": -0.5, "prediction": -1.0}])
    feat_df = pd.DataFrame(feat_rows)
    w_df = pd.DataFrame(weight_rows)
    bt, metrics, _ = backtest_long_short(feat_df, w_df, initial_capital=100.0, transaction_cost_bps=0.0)
    assert bt["turnover"].iloc[0] == 1.0
    assert bt["turnover"].iloc[1:].sum() == 0.0
    assert metrics["total_return"] > 0
