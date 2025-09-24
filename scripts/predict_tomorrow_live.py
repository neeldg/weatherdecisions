# scripts/predict_tomorrow_live.py
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import pytz

import openmeteo_requests
import requests_cache
from retry_requests import retry

# ------------------------------
# Config
# ------------------------------
LAT, LON = 30.2672, -97.7431
TZ = "America/Chicago"
DATA_CSV = "data/openmeteo_archive_austin_2000_2025.csv"
MODEL_TMAX = "models/xgb_tmax_D1_austin.pkl"
MODEL_TMIN = "models/xgb_tmin_D1_austin.pkl"
FEATURE_LIST = "models/xgb_temp_features.txt"  # for safety

# ------------------------------
# 1) Load daily history & build today's feature row
# ------------------------------
df = pd.read_csv(DATA_CSV)
if "date" not in df.columns:
    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    else:
        df = df.rename(columns={df.columns[0]: "date"})

# Parse to UTC then convert to local tz (handles mixed offsets)
dt_utc = pd.to_datetime(df["date"], utc=True, errors="raise")
df["date"] = dt_utc.dt.tz_convert(TZ)
df = df.set_index("date").sort_index()

daily = (df
         .resample("D")
         .agg({
             "temperature_2m": ["mean", "max", "min"],
             "relative_humidity_2m": "mean",
             "dew_point_2m": "mean",
             "wind_speed_10m": "mean",
             "precipitation": "sum"
         }))

# Flatten
daily.columns = ["_".join(c).strip() for c in daily.columns]
daily = daily.reset_index().rename(columns={
    "date": "date",
    "temperature_2m_mean": "tavg_f",
    "temperature_2m_max":  "tmax_f",
    "temperature_2m_min":  "tmin_f",
    "relative_humidity_2m_mean": "rh_mean",
    "dew_point_2m_mean":        "dew_mean",
    "wind_speed_10m_mean":      "wind_mean",
    "precipitation_sum":        "precip_sum"
})

# Build lags/rolls identical to training
for col in ["tavg_f","tmax_f","tmin_f","rh_mean","wind_mean","precip_sum"]:
    daily[f"{col.split('_')[0]}_lag1" if col!="precip_sum" else "precip_lag1"] = daily[col].shift(1)

for w in [3, 7]:
    daily[f"tavg_roll{w}"] = daily["tavg_f"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"tmax_roll{w}"] = daily["tmax_f"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"tmin_roll{w}"] = daily["tmin_f"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"rh_roll{w}"]   = daily["rh_mean"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"wind_roll{w}"] = daily["wind_mean"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"rain_roll{w}"] = daily["precip_sum"].rolling(w, min_periods=1).sum().shift(1)

daily["dow"] = daily["date"].dt.weekday
daily["doy"] = daily["date"].dt.dayofyear
daily["month"] = daily["date"].dt.month

# Feature list (must match training)
feature_cols = [
    "tavg_f","tmax_f","tmin_f","rh_mean","dew_mean","wind_mean","precip_sum",
    "tavg_lag1","tmax_lag1","tmin_lag1","rh_lag1","wind_lag1","precip_lag1",
    "tavg_roll3","tavg_roll7","tmax_roll3","tmax_roll7","tmin_roll3","tmin_roll7",
    "rh_roll3","rh_roll7","wind_roll3","wind_roll7","rain_roll3","rain_roll7",
    "dow","doy","month"
]
# (Optional) sanity: ensure they match saved list
if os.path.exists(FEATURE_LIST):
    saved = pd.read_csv(FEATURE_LIST, header=None)[0].tolist()
    assert feature_cols == saved, "Feature list mismatch; re-train or update features."

# Today in local tz
now_local = datetime.now(pytz.timezone(TZ))
today = pd.Timestamp(now_local.date(), tz=TZ)

# We need the last complete day for lags/rolls, i.e., yesterday
yesterday = today - pd.Timedelta(days=1)

# Ensure we have rows up to yesterday
daily = daily.dropna().sort_values("date")
assert daily["date"].max() >= yesterday, "Archive CSV is missing yesterday; refresh your archive if needed."

# Build the single feature row using yesterday’s row (no leakage)
row = daily[daily["date"] == yesterday].copy()
assert len(row) == 1, f"Expected one row for {yesterday.date()}, got {len(row)}"
Xrow = row[feature_cols].values

# ------------------------------
# 2) Load models & predict tomorrow (D+1)
# ------------------------------
m_tmax = joblib.load(MODEL_TMAX)
m_tmin = joblib.load(MODEL_TMIN)

pred_tmax = float(m_tmax.predict(Xrow)[0])
pred_tmin = float(m_tmin.predict(Xrow)[0])

target_date = (yesterday + pd.Timedelta(days=1)).date()  # D+1
print(f"Model D+1 for {target_date}: Tmax={pred_tmax:.2f}F  Tmin={pred_tmin:.2f}F")

# ------------------------------
# 3) Pull LIVE forecast (reference) and aggregate to daily
# ------------------------------
cache = requests_cache.CachedSession(".cache", expire_after=600)  # 10 min cache
session = retry(cache, retries=3, backoff_factor=0.2)
om = openmeteo_requests.Client(session=session)

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"],
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
    "timezone": TZ
}
resp = om.weather_api(url, params=params)[0]
hourly = resp.Hourly()

h_index = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
).tz_convert(TZ)

hdf = pd.DataFrame({
    "date": h_index,
    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
    "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
    "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
    "precipitation": hourly.Variables(3).ValuesAsNumpy(),
}).set_index("date")

dailyr = (hdf
          .resample("D")
          .agg({
              "temperature_2m": ["mean", "max", "min"],
              "relative_humidity_2m": "mean",
              "wind_speed_10m": "mean",
              "precipitation": "sum"
          }))
dailyr.columns = ["_".join(c).strip() for c in dailyr.columns]
dailyr = dailyr.reset_index().rename(columns={
    "date": "valid_date",
    "temperature_2m_mean": "f_tavg",
    "temperature_2m_max":  "f_tmax",
    "temperature_2m_min":  "f_tmin",
    "relative_humidity_2m_mean": "f_rh",
    "wind_speed_10m_mean": "f_wind",
    "precipitation_sum": "f_prcp"
})

frow = dailyr[dailyr["valid_date"].dt.date == target_date]
f_tmax = float(frow["f_tmax"].iloc[0]) if not frow.empty else np.nan
f_tmin = float(frow["f_tmin"].iloc[0]) if not frow.empty else np.nan

print(f"Open-Meteo D+1 for {target_date}: f_tmax={f_tmax:.2f}F  f_tmin={f_tmin:.2f}F")

# ------------------------------
# 4) Save result
# ------------------------------
os.makedirs("models", exist_ok=True)
out = pd.DataFrame([{
    "run_time_local": now_local.strftime("%Y-%m-%d %H:%M %Z"),
    "target_date": target_date.strftime("%Y-%m-%d"),
    "model_tmax": pred_tmax,
    "model_tmin": pred_tmin,
    "om_f_tmax": f_tmax,
    "om_f_tmin": f_tmin
}])
out_path = "models/tomorrow_temp_pred.csv"
out.to_csv(out_path, index=False)
print("Saved:", out_path)
