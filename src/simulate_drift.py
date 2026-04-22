"""
src/simulate_drift.py
---------------------
Generates a synthetic "current production" dataset by introducing
controlled drift into the reference dataset.

Use this to TEST your monitor.py works correctly before deploying.

Usage:
    python src/simulate_drift.py
    python src/simulate_drift.py --input data/churn.csv --drift high
"""

import argparse
import os

import numpy as np
import pandas as pd

OUTPUT_PATH = "data/current.csv"


def simulate_drift(df: pd.DataFrame, drift_level: str = "medium") -> pd.DataFrame:
    """
    Introduce realistic drift into a copy of the dataframe.

    drift_level:
        'low'    — subtle shift in 1-2 features
        'medium' — moderate shift in ~30% of features
        'high'   — strong drift in 50%+ of features (will trigger alerts)
    """
    current = df.copy()
    rng     = np.random.default_rng(seed=99)

    num_cols = current.select_dtypes(include="number").columns.tolist()
    n_cols   = len(num_cols)

    if drift_level == "low":
        drift_cols = num_cols[:max(1, n_cols // 5)]
        shift_factor = 0.15
    elif drift_level == "medium":
        drift_cols = num_cols[:max(2, n_cols // 3)]
        shift_factor = 0.35
    else:  # high
        drift_cols = num_cols[:max(3, n_cols // 2)]
        shift_factor = 0.70

    for col in drift_cols:
        col_std  = current[col].std()
        col_mean = current[col].mean()
        # Shift mean + add noise
        current[col] = current[col] + (col_mean * shift_factor) + rng.normal(0, col_std * 0.1, len(current))
        # Clip to avoid unrealistic negatives for non-negative columns
        if current[col].min() >= 0:
            current[col] = current[col].clip(lower=0)

    # Introduce some missing values (data quality drift)
    missing_cols = num_cols[:2]
    for col in missing_cols:
        mask = rng.random(len(current)) < 0.05   # 5% missing
        current.loc[mask, col] = np.nan

    print(f"✅ Drift simulation complete ({drift_level} level)")
    print(f"   Drifted columns : {drift_cols}")
    print(f"   Shape           : {current.shape}")
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/churn.csv")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--drift", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of rows to sample (default: all)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    df = pd.read_csv(args.input)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)

    current = simulate_drift(df, drift_level=args.drift)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    current.to_csv(args.output, index=False)
    print(f"   Saved to        : {args.output}")
    print(f"\n👉  Now run: python src/monitor.py --ref {args.input} --cur {args.output}")


if __name__ == "__main__":
    main()