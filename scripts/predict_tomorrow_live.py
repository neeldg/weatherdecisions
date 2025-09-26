import argparse
import pandas as pd
import joblib
from pathlib import Path

from features import build_latest_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/openmeteo_archive_austin_2000_2025.csv")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--target", default="temperature_2m")
    ap.add_argument("--model", default="models/xgb_temp2m_tomorrow.pkl")
    ap.add_argument("--features", default="models/xgb_features.txt")
    args = ap.parse_args()

    hist = pd.read_csv(args.data)
    with open(args.features) as f:
        FEATS = [ln.strip() for ln in f if ln.strip()]

    X_last = build_latest_features(hist, args.date_col, FEATS, args.target)
    model = joblib.load(args.model)
    yhat = model.predict(X_last)[0]
    print(f"Predicted tomorrow {args.target}: {yhat:.3f}")

if __name__ == "__main__":
    main()
