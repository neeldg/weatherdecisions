#!/usr/bin/env python3
# scripts/predict_with_alignment_fix.py

import sys
from pathlib import Path
# Ensure the repo root is on sys.path so `import scripts` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#!/usr/bin/env python3
# scripts/predict_with_alignment_fix.py
import pickle
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(".").resolve()
MODEL_PATH = REPO_ROOT / "models" / "xgb_tmax_D1_austin.pkl"   # change to xgb_tmin_D1_austin.pkl if needed
DATA_PATH = REPO_ROOT / "data" / "openmeteo_archive_austin_daily_2000_2025.csv"
FEATURES_PATH = REPO_ROOT / "models" / "xgb_temp_features.txt"

# ensure running from repo root
print("Repo root:", REPO_ROOT)

# Import features module from scripts package
# If running from repo root, 'scripts' should be importable
try:
    from scripts import features as featmod
except Exception as e:
    print("Failed to import scripts.features:", e)
    print("Make sure you run this from the repo root directory.")
    sys.exit(1)

# 1) Load model
print("Loading model:", MODEL_PATH)
with open(MODEL_PATH, "rb") as fh:
    model = pickle.load(fh)
print("Loaded model type:", type(model))

# 2) Load history CSV
print("Loading data:", DATA_PATH)
hist = pd.read_csv(DATA_PATH, parse_dates=["date"])

# 3) Load feature list from file into a python list (drop leading '0' if present)
feat_lines = [ln.strip() for ln in open(FEATURES_PATH, "r", encoding="utf-8").read().splitlines() if ln.strip()]
if feat_lines and feat_lines[0] == "0":
    feat_lines = feat_lines[1:]
feature_cols = feat_lines
print(f"Loaded {len(feature_cols)} feature names from {FEATURES_PATH}")

# 4) Call build_latest_features with a list of feature names (the function expects that)
date_col = "date"
target_col = "tmax"

print("Calling features.build_latest_features(history_df, date_col, feature_cols_list, target_col)...")
try:
    result = featmod.build_latest_features(hist, date_col, feature_cols, target_col)
except TypeError as te:
    # If signature different, try alternative orderings / attempts
    print("TypeError calling build_latest_features (trying alternate signatures):", te)
    try:
        result = featmod.build_latest_features(hist, date_col, target_col)
    except Exception as e:
        print("Fallback also failed:", e)
        raise
except Exception as e:
    print("Error while building features:", e)
    raise

# build_latest_features may return (df_feat, feats) or df only
if isinstance(result, tuple):
    df_feat = result[0]
else:
    df_feat = result

if df_feat is None or len(df_feat) == 0:
    print("build_latest_features returned no rows; aborting.")
    sys.exit(1)

# Keep last row only
df_row = df_feat.tail(1).copy()

print("Feature DataFrame shape produced:", df_row.shape)
print("Produced columns (sample):", list(df_row.columns)[:40])

# 5) Compare to expected feature list
expected = feature_cols
produced = list(df_row.columns)
extra = [c for c in produced if c not in expected]
missing = [c for c in expected if c not in produced]

print("Model expects %d features; produced %d features" % (len(expected), len(produced)))
print("Extra produced columns (will drop):", extra)
print("Missing expected columns (will add as NaN):", missing)

# 6) Align features (drop extras, add missing as NaN, reorder)
X_aligned = df_row.copy()
if extra:
    X_aligned = X_aligned.drop(columns=extra, errors="ignore")
for c in missing:
    X_aligned[c] = pd.NA
X_aligned = X_aligned.reindex(columns=expected)

# convert object dtypes to numeric where possible
for col in X_aligned.columns:
    if pd.api.types.is_object_dtype(X_aligned[col].dtype):
        X_aligned[col] = pd.to_numeric(X_aligned[col], errors="coerce")

print("Aligned features shape:", X_aligned.shape)
print("Aligned features (first 30):", X_aligned.columns.tolist()[:30])

# 7) Predict
print("Running model.predict on aligned features...")
try:
    yhat = model.predict(X_aligned)
    print("Raw prediction array:", yhat)
    # print last element (the next-day)
    val = float(yhat[-1]) if hasattr(yhat, "__len__") else float(yhat)
    print(f"Predicted next-day {target_col}:", val)
except Exception as e:
    print("Prediction failed. Exception:")
    import traceback
    traceback.print_exc()
    print("Model.feature_names_in_ (if available):", getattr(model, "feature_names_in_", None))
    sys.exit(1)
