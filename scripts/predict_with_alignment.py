#!/usr/bin/env python3
# scripts/predict_with_alignment.py
import pickle
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(".")
MODEL_PATH = REPO_ROOT / "models" / "xgb_tmax_D1_austin.pkl"   # change if you want tmin
DATA_PATH = REPO_ROOT / "data" / "openmeteo_archive_austin_daily_2000_2025.csv"
FEATURES_PATH = REPO_ROOT / "models" / "xgb_temp_features.txt"

# import the repo's features builder
# note: when running from repo root, the scripts directory is importable as a module
try:
    from scripts import features as featmod
except Exception as e:
    print("Failed to import scripts.features:", e)
    print("Make sure you run this from the repo root and your PYTHONPATH includes the repo root.")
    sys.exit(1)

# 1) Load model
print("Loading model:", MODEL_PATH)
with open(MODEL_PATH, "rb") as fh:
    model = pickle.load(fh)
print("Loaded model type:", type(model))

# 2) Load history CSV
print("Loading data:", DATA_PATH)
hist = pd.read_csv(DATA_PATH, parse_dates=[0])  # date parsing; features.build_latest_features will re-parse if needed

# 3) Build features using repo helper (signature: build_latest_features(history_df, date_col, feats_file_or_list, target_col))
# The repo's predict script used build_latest_features(hist, args.date_col, FEATS, args.target)
date_col = "date"
target_col = "tmax"
FEATS = str(FEATURES_PATH)  # feed same features file path as the repo does

print("Calling features.build_latest_features(...) to produce X_last")
X_last = None
try:
    X_last = featmod.build_latest_features(hist, date_col, FEATS, target_col)
    # build_latest_features may return (df_feat, feats) or just df; check:
    if isinstance(X_last, tuple) and len(X_last) >= 1:
        df_feat = X_last[0]
    else:
        df_feat = X_last
except TypeError as te:
    # maybe different signature: try build_latest_features(hist, date_col, target_col) fallback
    print("build_latest_features signature mismatch, trying fallback...")
    try:
        df_feat = featmod.build_latest_features(hist, date_col, target_col)
    except Exception as e:
        print("Fallback failed:", e)
        raise
except Exception as e:
    print("Error building features:", e)
    raise

if df_feat is None or len(df_feat) == 0:
    print("No features returned by build_latest_features. Inspect script/features.py for usage.")
    sys.exit(1)

# Keep only the last row (the one to predict)
if df_feat.shape[0] > 1:
    df_row = df_feat.tail(1).copy()
else:
    df_row = df_feat.copy()

print("Produced feature DataFrame shape:", df_row.shape)
print("Columns produced (sample):", list(df_row.columns))

# 4) Read expected feature list
feat_lines = [ln.strip() for ln in open(FEATURES_PATH, "r", encoding="utf-8").read().splitlines() if ln.strip()]
# Some feature files include a leading '0' line — drop it if present
if feat_lines and feat_lines[0] == "0":
    feat_lines = feat_lines[1:]
expected = feat_lines
print("Model expects %d features (sample): %s" % (len(expected), expected[:10]))

# 5) Compare
produced = list(df_row.columns)
extra = [c for c in produced if c not in expected]
missing = [c for c in expected if c not in produced]
print("Extra columns produced (will be dropped):", extra)
print("Missing columns expected by model (will be created as NaN):", missing)

# 6) Align: drop extras, add missing as NaN, reorder to expected
# Create aligned DataFrame X_aligned with columns in the expected order
X_aligned = df_row.copy()

# Drop extras (but keep a copy for debugging)
if extra:
    print("Dropping extra columns:", extra)
    X_aligned = X_aligned.drop(columns=extra, errors="ignore")

# Add missing columns as NaN
for c in missing:
    X_aligned[c] = pd.NA

# Reorder columns to expected
X_aligned = X_aligned.reindex(columns=expected)

print("Aligned feature DataFrame shape:", X_aligned.shape)
print("Aligned columns (first 30):", X_aligned.columns.tolist()[:30])

# 7) Convert dtypes if necessary (XGBoost wants numeric types)
for col in X_aligned.columns:
    if pd.api.types.is_object_dtype(X_aligned[col].dtype):
        try:
            X_aligned[col] = pd.to_numeric(X_aligned[col], errors="coerce")
        except Exception:
            pass

# 8) Predict
print("Calling model.predict(...)")
try:
    yhat = model.predict(X_aligned)
    print("Prediction array:", yhat)
    if hasattr(yhat, "__len__"):
        print("Predicted value for next-day:", float(yhat[-1]))
    else:
        print("Predicted scalar:", float(yhat))
except Exception as e:
    print("Predict failed with exception:")
    import traceback
    traceback.print_exc()
    # helpful diagnostics: print model.feature_names_in_ if available
    try:
        print("model.feature_names_in_:", getattr(model, "feature_names_in_", None))
    except Exception:
        pass
    sys.exit(1)

