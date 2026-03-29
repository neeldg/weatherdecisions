#!/usr/bin/env python3
"""
Run the trained XGBoost models to get next-day high (and low) temperature, then feed mu = pred_tmax
into the Gaussian fair-value model (normal CDF) for Kalshi-style temperature contracts.

This is the glue between "what our ML model predicts" and "what the pricing model uses."

Usage (from repo root):
    python scripts/predict_to_fair_value.py
    python scripts/predict_to_fair_value.py --sigma-high 2.5

Requires data/openmeteo_archive_austin_daily_2000_2025.csv and model pickles under models/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fair_value import (
    fair_value_cents,
    prob_high_geq,
    prob_high_in_range,
)
from scripts.daily_evaluate import (
    MODEL_TMAX,
    MODEL_TMIN,
    build_features_and_predict,
    read_daily_csv,
    DAILY_CSV,
    DATE_COL,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="XGBoost temp prediction → normal-model fair value (¢)")
    ap.add_argument("--daily-csv", type=Path, default=DAILY_CSV, help="Daily history CSV")
    ap.add_argument(
        "--sigma-high",
        type=float,
        default=3.0,
        help="Std dev (°F) for daily high under the Gaussian fair-value model. "
        "Tune to ensemble spread or historical forecast error.",
    )
    args = ap.parse_args()

    daily_df = read_daily_csv(Path(args.daily_csv))
    if daily_df is None:
        print("Daily CSV not found:", args.daily_csv, file=sys.stderr)
        sys.exit(1)

    pred_tmax = build_features_and_predict(MODEL_TMAX, daily_df, "tmax")
    pred_tmin = build_features_and_predict(MODEL_TMIN, daily_df, "tmin")
    mu = pred_tmax
    sig = args.sigma_high

    print("Model predictions (next day from latest history row)")
    print(f"  pred_tmax (μ for high-T contracts): {pred_tmax:.3f} °F")
    print(f"  pred_tmin: {pred_tmin:.3f} °F")
    print(f"  σ_high for Normal CDF: {sig:.3f} °F")
    print()

    rows = [
        ("HIGH 76–77°F", prob_high_in_range(mu, sig, 76.0, 77.0)),
        ("HIGH ≥80°F", prob_high_geq(mu, sig, 80.0)),
    ]

    print("Fair value from Gaussian model (μ = pred_tmax, σ = --sigma-high)")
    print(f"{'Contract':<18} {'P(event)':>10} {'Fair ¢':>8}")
    for name, p in rows:
        print(f"{name:<18} {p:10.4f} {fair_value_cents(p):8d}")

    print()
    print(
        "Precipitation, wind, and HDD/CDD contracts need separate likelihood models; "
        "they are not determined by the daily-high Normal alone."
    )


if __name__ == "__main__":
    main()
