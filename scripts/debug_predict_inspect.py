#!/usr/bin/env python3
"""
Diagnostic script to inspect features and model behavior.

- Builds the full feature matrix using repository feature builder
- Prints the single-row (next-day) feature values fed to the model
- Prints historical summary stats for each feature (mean/std/min/max/nans)
- Prints model info (type, get_params keys, feature_names_in_ if available)
- Prints feature importances (sklearn wrapper .feature_importances_ if present)
- Runs a quick historical prediction across the days where features are available
  and computes MAE/RMSE to show historical performance.
"""
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import traceback

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))   # ensure 'scripts' package is importable
from scripts import features as featmod

MODEL_PATH = REPO / "models" / "xgb_tmax_D1_austin.pkl"
DAILY_CSV = REPO / "data" / "openmeteo_archive_austin_daily_2000_2025.csv"
FEATURES_TXT = REPO / "models" / "xgb_temp_features.txt"
DATE_COL = "date"
TARGET = "tmax"

def load_feature_list(path):
    lines = [ln.strip() for ln in open(path, "r", encoding="utf-8").read().splitlines() if ln.strip()]
    if lines and lines[0] == "0":
        lines = lines[1:]
    return lines

def main():
    print("Repo:", REPO)
    # load model
    print("\nLoading model:", MODEL_PATH)
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    print("Model type:", type(model))
    try:
        print("Model get_params keys sample:", list(model.get_params().keys())[:20])
    except Exception as e:
        print("Could not call get_params():", e)
    print("Model.feature_names_in_:", getattr(model, "feature_names_in_", None))

    # load daily csv
    print("\nLoading daily CSV:", DAILY_CSV)
    df = pd.read_csv(DAILY_CSV, parse_dates=[DATE_COL])
    print("Daily rows:", len(df), "date range:", df[DATE_COL].min(), "->", df[DATE_COL].max())

    # read expected feature list
    feature_list = load_feature_list(FEATURES_TXT)
    print("Model expects %d features (sample): %s" % (len(feature_list), feature_list[:8]))

    # Build full features matrix (use build_features if available)
    print("\nBuilding features for full history using features.build_features(...)")
    try:
        df_feats, feats = featmod.build_features(df.copy(), DATE_COL, TARGET)
    except TypeError:
        # some versions may return just df_feats
        df_feats = featmod.build_features(df.copy(), DATE_COL, TARGET)
        feats = feature_list
    except Exception as e:
        print("Failed to run build_features():")
        traceback.print_exc()
        return

    print("Feature matrix shape:", df_feats.shape)
    # Ensure columns exist
    missing_cols = [c for c in feature_list if c not in df_feats.columns]
    if missing_cols:
        print("WARNING - the following expected columns are missing from produced features:", missing_cols)

    # Keep rows where model target isn't NaN (where build_features defined labels)
    # df_feats likely contains target shifted; find rows we can predict (non-null of feature columns)
    # We'll attempt to produce row-by-row predictions aligned to df_feats indices where next-day target exists
    # For single next-day, get last row
    X_last = df_feats.tail(1).copy()
    print("\n--- NEXT-DAY FEATURE VALUES (last row) ---")
    # Show each feature and its value
    pd.set_option("display.max_rows", 200)
    print(X_last[feature_list].T)

    # Historical feature stats
    print("\n--- HISTORICAL FEATURE SUMMARY (mean / std / min / max / n_null) ---")
    stats = []
    for c in feature_list:
        if c in df_feats.columns:
            col = pd.to_numeric(df_feats[c], errors="coerce")
            stats.append({
                "feature": c,
                "mean": float(col.mean()) if col.notna().any() else np.nan,
                "std": float(col.std()) if col.notna().any() else np.nan,
                "min": float(col.min()) if col.notna().any() else np.nan,
                "max": float(col.max()) if col.notna().any() else np.nan,
                "n_null": int(col.isna().sum())
            })
        else:
            stats.append({"feature": c, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n_null": None})
    stats_df = pd.DataFrame(stats).set_index("feature")
    print(stats_df.head(40).to_string())

    # Print model importances if available
    print("\n--- MODEL IMPORTANCE ---")
    try:
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=getattr(model, "feature_names_in_", feature_list))
            print(fi.sort_values(ascending=False).head(20).to_string())
        else:
            print("No feature_importances_ attribute available for model.")
    except Exception as e:
        print("Error retrieving importances:", e)

    # Quick historical backtest: predict for all rows where we can build features (skip NaNs)
    print("\n--- QUICK HISTORICAL BACKTEST (apply model to rows with no NaN in feature_list) ---")
    X_hist = df_feats[feature_list].copy()
    # convert to numeric, drop rows with all-NaN or where any feature is NaN
    X_hist = X_hist.apply(pd.to_numeric, errors="coerce")
    # create mask of rows where target (next-day) exists in original df if available:
    # If build_features created target column named TARGET, use it to align; else assume X_hist index aligns
    y_true = None
    if TARGET in df_feats.columns:
        y_true = df_feats[TARGET]
    else:
        # try to pull from df by shifting
        if TARGET in df.columns:
            y_true = df[TARGET].shift(-1)  # next day true
    # drop rows where any required feature is null
    valid_mask = X_hist.notna().all(axis=1)
    X_valid = X_hist[valid_mask]
    if y_true is not None:
        y_valid = pd.to_numeric(y_true, errors="coerce")[valid_mask]
        # drop rows where y_valid is nan
        both_valid = y_valid.notna()
        X_valid = X_valid[both_valid]
        y_valid = y_valid[both_valid]
    else:
        y_valid = None

    print("Rows with complete features:", len(X_valid))
    if len(X_valid) == 0:
        print("No rows with complete features - cannot backtest.")
    else:
        try:
            yhat_hist = model.predict(X_valid)
            print("Predicted sample (tail):", yhat_hist[-5:])
            if y_valid is not None:
                err = yhat_hist - y_valid.values
                mae = np.mean(np.abs(err))
                rmse = np.sqrt(np.mean(err**2))
                bias = np.mean(err)
                print(f"Historical quick-backtest (n={len(y_valid)}): MAE={mae:.3f}, RMSE={rmse:.3f}, bias={bias:.3f}")
                # show some cases where error large
                df_comp = pd.DataFrame({"y_true": y_valid.values, "yhat": yhat_hist, "err": yhat_hist - y_valid.values})
                print("\nTop 10 largest absolute errors (historic):")
                print(df_comp.assign(abs_err=df_comp["err"].abs()).sort_values("abs_err", ascending=False).head(10).to_string(index=False))
            else:
                print("Model predictions available but no y_true to compare against.")
        except Exception as e:
            print("Error while predicting historical rows:")
            traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()

