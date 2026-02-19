
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precog_trading.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Precog trading pipeline.")
    parser.add_argument("--data", type=str, default=None, help="Path to daily_prices.csv")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config" / "default_config.json"))
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate and run on synthetic data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(data_path=args.data, project_root=PROJECT_ROOT, config_path=args.config, generate_synthetic=args.generate_synthetic)
    print(summary)


if __name__ == "__main__":
    main()
