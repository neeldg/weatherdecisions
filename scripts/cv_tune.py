import argparse, json
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from itertools import product

from features import build_features
from split_utils import grouped_kfold_indices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/openmeteo_archive_austin_2000_2025.csv")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--target", default="temperature_2m")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    df, feats = build_features(raw, args.date_col, args.target)

    search_space = {
        "n_estimators": [400, 800, 1200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_lambda": [1.0, 2.0, 4.0],
    }

    combos = list(product(*search_space.values()))
    keys = list(search_space.keys())

    folds = grouped_kfold_indices(df, args.date_col, n_splits=args.folds)

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        maes = []
        for tr_idx, va_idx in folds:
            Xtr, ytr = df.iloc[tr_idx][feats], df.iloc[tr_idx][args.target]
            Xva, yva = df.iloc[va_idx][feats], df.iloc[va_idx][args.target]
            model = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                random_state=args.seed,
                n_jobs=-1,
            )
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = model.predict(Xva)
            maes.append(mean_absolute_error(yva, pred))
        results.append({"params": params, "cv_mae_mean": float(sum(maes)/len(maes))})

    results = sorted(results, key=lambda r: r["cv_mae_mean"])
    Path("models").mkdir(exist_ok=True)
    with open("models/cv_results.json","w") as f:
        json.dump(results, f, indent=2)
    best = results[0]
    print("Best params:", best["params"], "| CV MAE:", round(best["cv_mae_mean"], 3))

if __name__ == "__main__":
    main()
