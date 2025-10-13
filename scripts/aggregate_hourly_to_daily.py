#!/usr/bin/env python3
"""
scripts/aggregate_hourly_to_daily.py

Robust aggregator:
- Reads hourly Open-Meteo CSV (data/openmeteo_archive_austin_2000_2025.csv)
- Auto-detects units for temperature-like columns (Celsius, Fahrenheit, or double-converted)
- Converts to Fahrenheit if needed, or undoes a double-conversion
- Aggregates to daily (Austin local time, America/Chicago)
- Produces these daily columns (and saves to data/openmeteo_archive_austin_daily_2000_2025.csv):
    date (YYYY-MM-DD), tavg, tmax, tmin, rh_mean, dew_mean, wind_mean, precip_sum,
    tavg_f, tmax_f, tmin_f  (explicit Fahrenheit columns for the model)
- Makes a safe backup of previous daily file if present
"""
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import sys

# Config
IN_PATH = Path("data/openmeteo_archive_austin_2000_2025.csv")
OUT_PATH = Path("data/openmeteo_archive_austin_daily_2000_2025.csv")
BACKUP_SUFFIX = ".bak"
LOCAL_TZ = "America/Chicago"  # Austin

def backup(path: Path):
    if path.exists():
        bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        shutil.copy2(path, bak)
        print(f"Backup written: {bak}")

def detect_and_fix_temp_scale(df: pd.DataFrame, temp_candidates):
    """
    Heuristics:
     - median <= 60  -> likely Celsius (convert to F)
     - median between 60..120 -> likely Fahrenheit (no change)
     - median >= 120 -> likely double-converted -> invert (F_true = (F_wrong - 32) / 1.8)
    Applies transform in-place and returns a dict summarizing actions.
    """
    info = {}
    if not temp_candidates:
        return info
    # pick first available candidate as representative
    rep = temp_candidates[0]
    med = pd.to_numeric(df[rep], errors="coerce").median()
    info["rep_column"] = rep
    info["median"] = float(med) if not pd.isna(med) else None
    action = "none"
    if med is None or np.isnan(med):
        action = "no_data"
    elif med <= 60:
        action = "c_to_f"   # convert Celsius to Fahrenheit
        for c in temp_candidates:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") * 9.0/5.0 + 32.0
    elif med >= 120:
        action = "undo_double"  # undo double conversion
        for c in temp_candidates:
            if c in df.columns:
                df[c] = (pd.to_numeric(df[c], errors="coerce") - 32.0) / 1.8
    else:
        action = "assume_f"  # already Fahrenheit
    info["action"] = action
    return info

def find_temp_cols(df):
    # common hourly names in Open-Meteo: 'temperature_2m', 'dew_point_2m', 'apparent_temperature'
    temps = [c for c in df.columns if "temperature" in c or "apparent" in c or "dew_point" in c]
    # prefer the canonical temperature column 'temperature_2m' if present
    prioritized = []
    if "temperature_2m" in df.columns:
        prioritized.append("temperature_2m")
    # ensure dew_point and apparent included if present
    for c in ("dew_point_2m", "apparent_temperature"):
        if c in df.columns and c not in prioritized:
            prioritized.append(c)
    # then add others
    for c in temps:
        if c not in prioritized:
            prioritized.append(c)
    return prioritized

def main():
    if not IN_PATH.exists():
        print(f"Input hourly CSV not found at: {IN_PATH}")
        sys.exit(1)

    print(f"Reading hourly CSV: {IN_PATH} (this may take a moment) ...")
    # read without forcing parse to handle mixed tz; we'll parse explicitly
    df = pd.read_csv(IN_PATH, low_memory=False)

    # find temperature-like columns (temperature and dew/apparent)
    temp_candidates = find_temp_cols(df)
    print("Detected temperature-like columns (sample order):", temp_candidates[:6])

    # If no temperature columns, abort
    if not temp_candidates:
        print("No temperature columns found. Aborting.")
        sys.exit(1)

    # robust datetime parsing: parse as UTC where possible
    try:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    except Exception:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # auto-detect and fix scalings for temperature-like columns
    # We'll check the main temperature column candidate(s)
    temp_fix_info = detect_and_fix_temp_scale(df, temp_candidates)
    print("Temperature scale detection:", temp_fix_info)

    # Also ensure dew_point and apparent are scaled consistently if present:
    # For simplicity, apply the same action to dew_point/apparent cols if they exist
    if "action" in temp_fix_info and temp_fix_info["action"] in ("c_to_f","undo_double"):
        action = temp_fix_info["action"]
        for cname in ("dew_point_2m", "apparent_temperature"):
            if cname in df.columns:
                if action == "c_to_f":
                    df[cname] = pd.to_numeric(df[cname], errors="coerce") * 9.0/5.0 + 32.0
                elif action == "undo_double":
                    df[cname] = (pd.to_numeric(df[cname], errors="coerce") - 32.0) / 1.8

    # Now convert to local Austin time for daily grouping
    # If date is tz-aware we tz_convert, else assume it is local/naive and try to localize to UTC then convert
    try:
        if df["date"].dt.tz is None:
            # naive -> assume UTC then convert (most Open-Meteo saves tz info; but be defensive)
            df["date"] = df["date"].dt.tz_localize("UTC")
        df["local_dt"] = df["date"].dt.tz_convert(LOCAL_TZ)
    except Exception:
        # fallback: coerce to naive datetimes (best effort)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["local_dt"] = df["date"]

    # Extract local date (normalize to midnight) and drop tz info for grouping
    df["day"] = df["local_dt"].dt.normalize().dt.tz_localize(None)

    # Choose which column to treat as the main temperature (for tavg/tmax/tmin)
    # Prefer daily-aggregated candidates if already present, otherwise use 'temperature_2m'
    if "temperature_2m" in df.columns:
        temp_col = "temperature_2m"
    else:
        # fallback to first temperature-like column
        temp_col = temp_candidates[0]
    print("Using temperature column for aggregation:", temp_col)

    # choose humidity, dew, wind, precip columns if present
    rh_col = None
    for c in ("relative_humidity_2m", "rh_mean", "relative_humidity"):
        if c in df.columns:
            rh_col = c; break
    dew_col = None
    for c in ("dew_point_2m", "dew_mean", "dew_point"):
        if c in df.columns:
            dew_col = c; break
    wind_col = None
    for c in ("wind_speed_10m", "wind_mean", "wind_speed"):
        if c in df.columns:
            wind_col = c; break
    precip_col = None
    for c in ("precipitation", "precip_sum", "rain", "rain_sum"):
        if c in df.columns:
            precip_col = c; break

    print("Using columns -> humidity:", rh_col, "dew:", dew_col, "wind:", wind_col, "precip:", precip_col)

    # Aggregate per-day
    grouped = df.groupby("day", sort=True)
    daily = pd.DataFrame()
    daily["tavg"] = grouped[temp_col].mean()
    daily["tmax"] = grouped[temp_col].max()
    daily["tmin"] = grouped[temp_col].min()
    if rh_col:
        daily["rh_mean"] = grouped[rh_col].mean()
    else:
        daily["rh_mean"] = np.nan
    if dew_col:
        daily["dew_mean"] = grouped[dew_col].mean()
    else:
        # If dew not provided, attempt to approximate from tavg and rh if available
        # (simple, not perfect): dew ≈ tavg - (100 - rh_mean)/5  (approximate Magnus-based approx)
        if "rh_mean" in daily and daily["rh_mean"].notnull().any():
            daily["dew_mean"] = daily["tavg"] - (100.0 - daily["rh_mean"]) / 5.0
        else:
            daily["dew_mean"] = np.nan
    if wind_col:
        daily["wind_mean"] = grouped[wind_col].mean()
    else:
        daily["wind_mean"] = np.nan
    if precip_col:
        # precipitation typically hourly sum; daily sum is appropriate
        daily["precip_sum"] = grouped[precip_col].sum()
    else:
        daily["precip_sum"] = 0.0

    # Reset index and format date column
    daily = daily.reset_index().rename(columns={"day": "date"})
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    # Add explicit Fahrenheit columns expected by model (ensure not double converted)
    # The aggregated tavg/tmax/tmin are already in Fahrenheit based on earlier detection.
    daily["tavg_f"] = daily["tavg"]
    daily["tmax_f"] = daily["tmax"]
    daily["tmin_f"] = daily["tmin"]

    # Re-order columns to put date first
    cols_order = ["date", "tavg", "tmax", "tmin", "rh_mean", "dew_mean", "wind_mean", "precip_sum",
                  "tavg_f", "tmax_f", "tmin_f"]
    # add any extra cols safely
    for c in daily.columns:
        if c not in cols_order:
            cols_order.append(c)
    daily = daily.loc[:, cols_order]

    # Backup existing daily CSV if present
    backup(OUT_PATH)

    # Save to CSV
    daily.to_csv(OUT_PATH, index=False)
    print(f"Wrote daily CSV: {OUT_PATH}")
    print("Daily sample (tail):")
    print(daily.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()
