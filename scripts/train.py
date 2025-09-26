import argparse, json, hashlib
from pathlib import Path
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_features
from split_utils import split_train_val_test
from baselines import persistence_on_val

def md5(path):
    m = hashlib.md5()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/openmeteo_archive_austin_2000_2025.csv")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--target", default="temperature_2m")
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac",  type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    # If you ran cv_tune.py, paste best params here; otherwise use solid defaults:
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.9)
    ap.add_argument("--l2", type=float, default=2.0)
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    df, FEATS = build_features(raw, args.date_col, args.target)

    tr_idx, va_idx, te_idx = split_train_val_test(
        df, args.date_col, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed
    )

    Xtr, ytr = df.iloc[tr_idx][FEATS], df.iloc[tr_idx][args.target]
    Xva, yva = df.iloc[va_idx][FEATS], df.iloc[va_idx][args.target]
    Xte, yte = df.iloc[te_idx][FEATS], df.iloc[te_idx][args.target]

    # Baseline on validation
    base_val = persistence_on_val(df.iloc[va_idx], args.target)
    base_mae = mean_absolute_error(yva, base_val)

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        reg_lambda=args.l2,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    va_pred = model.predict(Xva)
    te_pred = model.predict(Xte)

    val_mae = mean_absolute_error(yva, va_pred)
    val_rmse = mean_squared_error(yva, va_pred, squared=False)
    test_mae = mean_absolute_error(yte, te_pred)
    test_rmse = mean_squared_error(yte, te_pred, squared=False)

    Path("models").mkdir(exist_ok=True)
    model_path = Path("models/xgb_temp2m_tomorrow.pkl")
    joblib.dump(model, model_path)
    with open("models/xgb_features.txt","w") as f:
        f.write("\n".join(FEATS))

    metrics = {
        "data_file": Path(args.data).name,
        "data_md5": md5(args.data),
        "target": args.target,
        "n_features": len(FEATS),
        "val": {"MAE": float(val_mae), "RMSE": float(val_rmse), "persistence_MAE": float(base_mae)},
        "test": {"MAE": float(test_mae), "RMSE": float(test_rmse)},
        "params": {
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "learning_rate": args.lr, "subsample": args.subsample,
            "colsample_bytree": args.colsample, "reg_lambda": args.l2,
            "seed": args.seed
        },
    }
    with open("models/metrics_temperature_2m.json","w") as f:
        json.dump(metrics, f, indent=2)

    print(f"VAL  MAE: {val_mae:.3f} | RMSE: {val_rmse:.3f} | Persistence MAE: {base_mae:.3f}")
    print(f"TEST MAE: {test_mae:.3f} | RMSE: {test_rmse:.3f}")
    print("Saved:", model_path)

if __name__ == "__main__":
    main()
