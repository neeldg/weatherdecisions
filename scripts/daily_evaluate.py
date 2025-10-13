#!/usr/bin/env python3
"""
scripts/daily_evaluate.py

1) Builds next-day tmax/tmin predictions using repo models and feature pipeline.
2) Looks up the actual observed tmax/tmin for that next-day:
   - First: checks local daily CSV (data/openmeteo_archive_austin_daily_2000_2025.csv)
   - If not present: queries Open-Meteo archive for the date (no API key needed).
3) Appends results to results/nextday_results.csv and prints summary metrics.

Run from repo root:
    python scripts/daily_evaluate.py
"""
import os
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import requests
from datetime import timedelta, datetime

# --- CONFIG (edit if you want different filenames / station coords) ---
REPO_ROOT = Path(".").resolve()
DAILY_CSV = REPO_ROOT / "data" / "openmeteo_archive_austin_daily_2000_2025.csv"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "nextday_results.csv"
FEATURES_FILE = REPO_ROOT / "models" / "xgb_temp_features.txt"
MODEL_TMAX = REPO_ROOT / "models" / "xgb_tmax_D1_austin.pkl"
MODEL_TMIN = REPO_ROOT / "models" / "xgb_tmin_D1_austin.pkl"
DATE_COL = "date"

# Austin coordinates for Open-Meteo fallback (lat, lon)
AUSTIN_LAT = 30.2672
AUSTIN_LON = -97.7431
OPENMETEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# --- utility helpers ---
def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_model(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)

def read_daily_csv(path):
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    # ensure date column is date-only (normalize)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
    return df

def fetch_observed_from_openmeteo(date_obj):
    """Fetch daily max/min from Open-Meteo archive for the given date (date_obj is datetime.date)."""
    s = date_obj.strftime("%Y-%m-%d")
    e = s
    params = {
        "latitude": AUSTIN_LAT,
        "longitude": AUSTIN_LON,
        "start_date": s,
        "end_date": e,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "America/Chicago"
    }
    r = requests.get(OPENMETEO_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    # Open-Meteo returns 'daily' with arrays; check safely
    daily = data.get("daily", {})
    try:
        tmax_arr = daily.get("temperature_2m_max")
        tmin_arr = daily.get("temperature_2m_min")
        if tmax_arr and tmin_arr:
            return float(tmax_arr[0]), float(tmin_arr[0])
    except Exception:
        pass
    return None, None

def build_features_and_predict(model_path, daily_df, target):
    """Uses scripts.features.build_latest_features to build the single-row feature DF and predict."""
    # Ensure repo 'scripts' is importable
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import features as featmod
    except Exception as e:
        raise RuntimeError("Failed to import scripts.features: run from repo root and ensure PYTHONPATH") from e

    # Load feature name list
    feat_lines = [ln.strip() for ln in open(FEATURES_FILE, "r", encoding="utf-8").read().splitlines() if ln.strip()]
    if feat_lines and feat_lines[0] == "0":
        feat_lines = feat_lines[1:]
    feature_cols = feat_lines

    # Build latest features (function signature in repo: build_latest_features(history_df, date_col, feature_cols, target_col))
    res = featmod.build_latest_features(daily_df, DATE_COL, feature_cols, target)
    if isinstance(res, tuple):
        df_feat = res[0]
    else:
        df_feat = res
    if df_feat is None or df_feat.shape[0] == 0:
        raise RuntimeError("build_latest_features returned no rows")

    # keep last row and align to expected feature order
    row = df_feat.tail(1).copy()
    # drop extras, add missing as NaN, reorder
    extra = [c for c in row.columns if c not in feature_cols]
    if extra:
        row = row.drop(columns=extra, errors="ignore")
    for c in feature_cols:
        if c not in row.columns:
            row[c] = pd.NA
    row = row.reindex(columns=feature_cols)
    # convert objects to numeric where possible
    for c in row.columns:
        if pd.api.types.is_object_dtype(row[c].dtype):
            row[c] = pd.to_numeric(row[c], errors="coerce")

    # load model and predict
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    y = model.predict(row)
    if hasattr(y, "__len__"):
        return float(y[-1])
    return float(y)

def append_result_row(rowdict):
    ensure_results_dir()
    df_row = pd.DataFrame([rowdict])
    if RESULTS_CSV.exists():
        df_existing = pd.read_csv(RESULTS_CSV, parse_dates=["date_pred_for"])
        df_out = pd.concat([df_existing, df_row], ignore_index=True, sort=False)
    else:
        df_out = df_row
    # write
    df_out.to_csv(RESULTS_CSV, index=False)
    return df_out

def compute_metrics(df):
    # assume df has pred_tmax, actual_tmax, pred_tmin, actual_tmin
    metrics = {}
    if "pred_tmax" in df.columns and "actual_tmax" in df.columns and df["actual_tmax"].notna().any():
        err = df["pred_tmax"] - df["actual_tmax"]
        metrics["tmax_mae"] = float(err.abs().mean())
        metrics["tmax_rmse"] = float((err**2).mean()**0.5)
        metrics["tmax_bias"] = float(err.mean())
    if "pred_tmin" in df.columns and "actual_tmin" in df.columns and df["actual_tmin"].notna().any():
        err = df["pred_tmin"] - df["actual_tmin"]
        metrics["tmin_mae"] = float(err.abs().mean())
        metrics["tmin_rmse"] = float((err**2).mean()**0.5)
        metrics["tmin_bias"] = float(err.mean())
    return metrics

# ------------------- main flow --------------------
def main():
    # 1) load daily CSV
    daily_df = read_daily_csv(DAILY_CSV)
    if daily_df is None:
        print("Daily CSV not found at", DAILY_CSV)
        print("Please create daily CSV (see scripts/aggregate_hourly_to_daily.py) and re-run.")
        sys.exit(1)

    # determine next-day to predict
    last_date = daily_df[DATE_COL].max()  # date object
    if isinstance(last_date, pd.Timestamp):
        last_date = last_date.date()
    next_date = last_date + timedelta(days=1)
    print("Last date in daily CSV:", last_date, "→ predicting for:", next_date)

    # 2) predict tmax and tmin using repo models
    try:
        pred_tmax = build_features_and_predict(MODEL_TMAX, daily_df, "tmax")
    except Exception as e:
        print("Failed to predict tmax:", e)
        raise
    try:
        pred_tmin = build_features_and_predict(MODEL_TMIN, daily_df, "tmin")
    except Exception as e:
        print("Failed to predict tmin:", e)
        raise

    print(f"Predicted for {next_date}: tmax={pred_tmax:.3f}, tmin={pred_tmin:.3f}")

    # 3) find actuals: check CSV first
    actual_row = daily_df[daily_df[DATE_COL] == pd.to_datetime(next_date).date()]
    if not actual_row.empty:
        actual_tmax = float(actual_row["tmax"].iloc[0]) if "tmax" in actual_row.columns else np.nan
        actual_tmin = float(actual_row["tmin"].iloc[0]) if "tmin" in actual_row.columns else np.nan
        source = "local_csv"
    else:
        # attempt Open-Meteo archive fetch
        print("No local actual found for", next_date, "- querying Open-Meteo archive...")
        try:
            tmax, tmin = fetch_observed_from_openmeteo(next_date)
            actual_tmax = tmax
            actual_tmin = tmin
            source = "openmeteo"
        except Exception as e:
            print("Failed to fetch observed values from Open-Meteo:", e)
            actual_tmax = None
            actual_tmin = None
            source = "none"

    # 4) append to results
    row = {
        "date_pred_for": next_date.isoformat(),
        "pred_tmax": pred_tmax,
        "pred_tmin": pred_tmin,
        "actual_tmax": actual_tmax if actual_tmax is not None else np.nan,
        "actual_tmin": actual_tmin if actual_tmin is not None else np.nan,
        "actual_source": source,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    df_all = append_result_row(row)
    print("Appended result to:", RESULTS_CSV)
    # 5) compute metrics and print summary
    metrics = compute_metrics(df_all)
    print("Summary metrics (on results CSV):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    print("Done.")

if __name__ == "__main__":
    main()
