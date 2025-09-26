import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from features import build_features
from split_utils import groups_by_day

def rolling_blocks(unique_days, k=5):
    """Split unique days into k chronological blocks."""
    blocks = np.array_split(unique_days, k)
    return [np.array(b) for b in blocks]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/openmeteo_archive_austin_2000_2025.csv")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--target", default="temperature_2m")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    df, feats = build_features(raw, args.date_col, args.target)

    days = pd.to_datetime(df[args.date_col]).dt.date
    unique_days = pd.Series(days).drop_duplicates().to_numpy()
    blocks = rolling_blocks(unique_days, k=args.k)

    maes = []
    # rolling-origin: train on blocks <= i, validate on block i+1
    for i in range(len(blocks)-1):
        train_days = np.concatenate(blocks[:i+1])
        val_days   = blocks[i+1]

        tr_mask = days.isin(train_days)
        va_mask = days.isin(val_days)

        Xtr, ytr = df.loc[tr_mask, feats], df.loc[tr_mask, args.target]
        Xva, yva = df.loc[va_mask, feats], df.loc[va_mask, args.target]

        model = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03, n_jobs=-1)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred = model.predict(Xva)
        maes.append(mean_absolute_error(yva, pred))
        print(f"Block {i+1}->{i+2} MAE: {maes[-1]:.3f}")

    print("Rolling-origin CV MAE (mean):", round(float(np.mean(maes)), 3))

if __name__ == "__main__":
    main()
