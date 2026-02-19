
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precog_trading.cleaning import clean_prices


def test_clean_prices_removes_duplicates_and_fills_short_gaps():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "ticker": ["AAA", "AAA", "AAA", "AAA", "AAA"],
        "open": [10, 10, None, 10.4, 10.5],
        "high": [10.2, 10.2, None, 10.6, 10.7],
        "low": [9.9, 9.9, None, 10.2, 10.3],
        "close": [10.1, 10.1, 10.2, 10.5, 10.6],
        "volume": [1000, 1000, None, 1200, 1100],
    })
    result = clean_prices(df, max_forward_fill_days=2)
    cleaned = result.data
    assert len(cleaned) == 4
    assert cleaned["open"].isna().sum() == 0
    assert result.report["duplicate_rows_removed"] == 1
