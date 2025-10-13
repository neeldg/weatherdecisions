#!/usr/bin/env python3
"""
Builds the 28 features expected by the model from your daily CSV (robustly),
aligns them to the model's feature-order, and runs the model to print the prediction.

This is a pragmatic, defensive implementation to get a correct next-day prediction now.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

REPO = Path(".").resolve()
DAILY = REPO / "data" / "openmeteo_archive_austin_daily_2000_2025.csv"
FEATURES_TXT = REPO / "models" / "xgb_temp_features.txt"
MODEL = REPO / "models" / "xgb_tmax_D1_austin.pkl"

def load_feature_list(path):
    lines = [ln.strip() for ln in open(path, "r", encoding="utf-8").read().splitlines() if ln.strip()]
    if lines and lines[0] == "0":
        lines = lines[1:]
    return lines

def choose_temp_col(df):
    # Prefer explicit tavg/tmax/tmin if present, otherwise attempt to synthesize from 'temperature_2m'
    if {'tavg','tmax','tmin'}.issubset(set(df.columns)):
        return "tavg","tmax","tmin"
    if 'temperature_2m' in df.columns:
        # if hourly->daily aggregator produced `temperature_2m` (already daily avg) treat as tavg proxy
        return "temperature_2m","temperature_2m","temperature_2m"
    # fallback lookups
    for c in ("tavg","tmax","tmin","temp","temp_mean"):
        if c in df.columns:
            return c,c,c
    raise RuntimeError("No temperature column found in daily CSV")

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def main():
    if not DAILY.exists():
        print("Daily CSV not found:", DAILY)
        sys.exit(1)

    df = pd.read_csv(DAILY, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # ensure date is date-only (no timezone)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    # choose base temperature columns
    tavg_col, tmax_col, tmin_col = choose_temp_col(df)
    print("Using temp columns:", tavg_col, tmax_col, tmin_col)

    # convert to numeric
    df[tavg_col] = safe_numeric(df[tavg_col])
    df[tmax_col] = safe_numeric(df[tmax_col])
    df[tmin_col] = safe_numeric(df[tmin_col])

    # humidity, dew, wind, precip candidates
    rh_col = None
    for c in ("rh_mean","relative_humidity_2m","relative_humidity"):
        if c in df.columns:
            rh_col = c; break
    dew_col = None
    for c in ("dew_mean","dew_point_2m","dew_point"):
        if c in df.columns:
            dew_col = c; break
    wind_col = None
    for c in ("wind_mean","wind_speed_10m","wind_speed"):
        if c in df.columns:
            wind_col = c; break
    precip_col = None
    for c in ("precip_sum","precipitation","rain_sum"):
        if c in df.columns:
            precip_col = c; break

    # coerce numeric
    if rh_col: df[rh_col] = safe_numeric(df[rh_col])
    if dew_col: df[dew_col] = safe_numeric(df[dew_col])
    if wind_col: df[wind_col] = safe_numeric(df[wind_col])
    if precip_col: df[precip_col] = safe_numeric(df[precip_col])

    # We'll construct the model feature names exactly:
    feats = load_feature_list(FEATURES_TXT)
    print("Model expects %d features" % len(feats))

    # Create a working DataFrame to compute intermediate values
    w = pd.DataFrame(index=df.index)
    # temperatures - we will create Fahrenheit-named columns with _f suffix as model expects
    w['tavg_f'] = df[tavg_col]
    w['tmax_f'] = df[tmax_col]
    w['tmin_f'] = df[tmin_col]

    # humidity/dew/wind/precip: map to expected names
    w['rh_mean'] = df[rh_col] if rh_col else np.nan
    w['dew_mean'] = df[dew_col] if dew_col else np.nan
    w['wind_mean'] = df[wind_col] if wind_col else np.nan
    w['precip_sum'] = df[precip_col] if precip_col else np.nan

    # compute lags: lag1 = previous day's value
    for s in ['tavg_f','tmax_f','tmin_f','rh_mean','wind_mean','precip_sum']:
        w[f"{s.split('_')[0]}_lag1"] = w[s].shift(1)

    # compute rolling means on previous days (3 and 7), exclude current day: roll(...).shift(1)
    # For tavg_roll3/tavg_roll7 etc use the base tavg_f, tmax_f, tmin_f
    w['tavg_roll3'] = w['tavg_f'].rolling(window=3, min_periods=1).mean().shift(1)
    w['tavg_roll7'] = w['tavg_f'].rolling(window=7, min_periods=1).mean().shift(1)
    w['tmax_roll3'] = w['tmax_f'].rolling(window=3, min_periods=1).mean().shift(1)
    w['tmax_roll7'] = w['tmax_f'].rolling(window=7, min_periods=1).mean().shift(1)
    w['tmin_roll3'] = w['tmin_f'].rolling(window=3, min_periods=1).mean().shift(1)
    w['tmin_roll7'] = w['tmin_f'].rolling(window=7, min_periods=1).mean().shift(1)

    # relative humidity rolls
    w['rh_roll3'] = w['rh_mean'].rolling(window=3, min_periods=1).mean().shift(1)
    w['rh_roll7'] = w['rh_mean'].rolling(window=7, min_periods=1).mean().shift(1)
    # wind rolls
    w['wind_roll3'] = w['wind_mean'].rolling(window=3, min_periods=1).mean().shift(1)
    w['wind_roll7'] = w['wind_mean'].rolling(window=7, min_periods=1).mean().shift(1)
    # rain/precip rolls
    w['rain_roll3'] = w['precip_sum'].rolling(window=3, min_periods=1).mean().shift(1)
    w['rain_roll7'] = w['precip_sum'].rolling(window=7, min_periods=1).mean().shift(1)

    # dow/doy/month from df['date']
    w['dow'] = pd.to_datetime(df['date']).dt.dayofweek  # Monday=0 ... Sunday=6
    w['doy'] = pd.to_datetime(df['date']).dt.dayofyear
    w['month'] = pd.to_datetime(df['date']).dt.month

    # Keep the same index and get last row
    final = w.tail(1).copy()

    # Report produced feature values
    print("\nProduced features (last row):")
    for c in feats:
        val = final[c].iloc[0] if c in final.columns else np.nan
        print(f"{c:15s} : {val}")

    # Align into feature order expected by model
    X = pd.DataFrame(columns=feats)
    for c in feats:
        if c in final.columns:
            X.loc[0,c] = final[c].iloc[0]
        else:
            X.loc[0,c] = np.nan

    # convert to numeric types
    X = X.apply(pd.to_numeric, errors='coerce')

    # Load model and predict
    with open(MODEL, "rb") as fh:
        m = pickle.load(fh)
    y = m.predict(X)
    print("\nModel prediction output array:", y)
    print("Predicted next-day tmax (manual features):", float(y[-1]))

if __name__ == "__main__":
    main()
