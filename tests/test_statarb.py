
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precog_trading.statarb import discover_pairs


def test_discover_pairs_finds_cointegrated_pair():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-01", periods=300)
    x = np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))) + 4.0)
    y = x * np.exp(rng.normal(0, 0.005, len(dates)))
    z = np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))) + 3.5)
    df = pd.DataFrame({"date": list(dates) * 3, "ticker": ["X"] * len(dates) + ["Y"] * len(dates) + ["Z"] * len(dates), "close": list(x) + list(y) + list(z)})
    pairs = discover_pairs(df, lookback_days=252, min_correlation=0.2, max_pvalue=0.05, top_k=5)
    assert not pairs.empty
    pair_names = {tuple(sorted((r["ticker_y"], r["ticker_x"]))) for _, r in pairs.iterrows()}
    assert ("X", "Y") in pair_names
