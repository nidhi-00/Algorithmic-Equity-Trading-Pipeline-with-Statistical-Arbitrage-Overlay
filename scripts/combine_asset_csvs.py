from pathlib import Path
import argparse
import pandas as pd

ALIASES = {
    "date": {"date", "datetime", "timestamp", "time"},
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "adj_close", "adjusted_close", "c"},
    "volume": {"volume", "vol", "v"},
}

def clean_col(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")

def normalise_columns(df):
    rename = {}
    for col in df.columns:
        cleaned = clean_col(col)
        target = cleaned
        for canonical, names in ALIASES.items():
            if cleaned in names:
                target = canonical
                break
        rename[col] = target
    return df.rename(columns=rename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/anonymized_data")
    parser.add_argument("--output", default="data/daily_prices.csv")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []

    for file in files:
        df = pd.read_csv(file)
        df = normalise_columns(df)

        if "date" not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "date"})

        df["ticker"] = file.stem

        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{file.name} is missing columns {missing}. Found: {list(df.columns)}")

        df = df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(combined)}")
    print(f"Tickers: {combined['ticker'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(combined.head())

if __name__ == "__main__":
    main()
