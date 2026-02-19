
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precog_trading.features import engineer_features


def test_engineer_features_creates_target_and_features():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "ticker": "AAA", "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i, "volume": 1000 + i})
    df = pd.DataFrame(rows)
    feat = engineer_features(df, horizon=5)
    assert "ret_20" in feat.columns
    assert "target_fwd_return" in feat.columns
    assert len(feat["target_fwd_return"].dropna()) > 0
