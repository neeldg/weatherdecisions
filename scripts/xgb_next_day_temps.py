# scripts/xgb_nextday_temp.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# ---------- 1) Robust load + make daily features ----------
path = "data/openmeteo_archive_austin_2000_2025.csv"
df = pd.read_csv(path)

# Ensure we have a 'date' column name
if "date" not in df.columns:
    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    else:
        df = df.rename(columns={df.columns[0]: "date"})

# Parse with UTC to handle mixed offsets, then convert to local tz via .dt
dt_utc = pd.to_datetime(df["date"], utc=True, errors="raise")
df["date"] = dt_utc.dt.tz_convert("America/Chicago")

# Use DatetimeIndex for resampling
df = df.set_index("date").sort_index()
print("Index dtype:", df.index.dtype)  # expect: datetime64[ns, America/Chicago]

daily = (df
         .resample("D")
         .agg({
             "temperature_2m": ["mean", "max", "min"],
             "relative_humidity_2m": "mean",
             "dew_point_2m": "mean",
             "wind_speed_10m": "mean",
             "precipitation": "sum"
         }))

# Flatten columns
daily.columns = ["_".join(col).strip() for col in daily.columns.values]
daily = daily.reset_index()

# Rename for clarity
daily = daily.rename(columns={
    "temperature_2m_mean": "tavg_f",
    "temperature_2m_max":  "tmax_f",
    "temperature_2m_min":  "tmin_f",
    "relative_humidity_2m_mean": "rh_mean",
    "dew_point_2m_mean":        "dew_mean",
    "wind_speed_10m_mean":      "wind_mean",
    "precipitation_sum":        "precip_sum"
})

# ---------- Rolling training window (last N years) ----------
WINDOW_YEARS = 10  # <- change here if you want 5, 7, etc.

last_date = daily["date"].max()
cutoff_start = (last_date - pd.DateOffset(years=WINDOW_YEARS))
# keep timezone; no normalize() so we don’t drop tz info
daily = daily[daily["date"] >= cutoff_start].copy()

print(f"Rolling window: {cutoff_start.date()} → {last_date.date()}   n_days={len(daily)}")

# ---------- Targets: next day's Tmax and Tmin ----------
daily["tmax_f_D1"] = daily["tmax_f"].shift(-1)
daily["tmin_f_D1"] = daily["tmin_f"].shift(-1)

# ---------- Lag & rolling features (no leakage) ----------
daily["tavg_lag1"]   = daily["tavg_f"].shift(1)
daily["tmax_lag1"]   = daily["tmax_f"].shift(1)
daily["tmin_lag1"]   = daily["tmin_f"].shift(1)
daily["rh_lag1"]     = daily["rh_mean"].shift(1)
daily["wind_lag1"]   = daily["wind_mean"].shift(1)
daily["precip_lag1"] = daily["precip_sum"].shift(1)

for w in [3, 7]:
    daily[f"tavg_roll{w}"] = daily["tavg_f"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"tmax_roll{w}"] = daily["tmax_f"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"tmin_roll{w}"] = daily["tmin_f"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"rh_roll{w}"]   = daily["rh_mean"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"wind_roll{w}"] = daily["wind_mean"].rolling(w, min_periods=1).mean().shift(1)
    daily[f"rain_roll{w}"] = daily["precip_sum"].rolling(w, min_periods=1).sum().shift(1)

# Calendar
daily["dow"] = daily["date"].dt.weekday
daily["doy"] = daily["date"].dt.dayofyear
daily["month"] = daily["date"].dt.month

# Drop rows with NaNs from shifts/rolls/targets
daily = daily.dropna().reset_index(drop=True)

# ---------- Feature set ----------
feature_cols = [
    "tavg_f","tmax_f","tmin_f","rh_mean","dew_mean","wind_mean","precip_sum",
    "tavg_lag1","tmax_lag1","tmin_lag1","rh_lag1","wind_lag1","precip_lag1",
    "tavg_roll3","tavg_roll7","tmax_roll3","tmax_roll7","tmin_roll3","tmin_roll7",
    "rh_roll3","rh_roll7","wind_roll3","wind_roll7","rain_roll3","rain_roll7",
    "dow","doy","month"
]

# ---------- Train/valid split by time (last 20% = validation) ----------
split_idx = int(len(daily) * 0.8)

def split_xy(target_col):
    X = daily[feature_cols].values
    y = daily[target_col].values
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

Xtr_tmax, ytr_tmax, Xva_tmax, yva_tmax = split_xy("tmax_f_D1")
Xtr_tmin, ytr_tmin, Xva_tmin, yva_tmin = split_xy("tmin_f_D1")

# ---------- Train two XGB models ----------
def make_model():
    return XGBRegressor(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42
    )

m_tmax = make_model().fit(Xtr_tmax, ytr_tmax, eval_set=[(Xva_tmax, yva_tmax)], verbose=False)
m_tmin = make_model().fit(Xtr_tmin, ytr_tmin, eval_set=[(Xva_tmin, yva_tmin)], verbose=False)

# ---------- Evaluate vs. persistence ----------
def eval_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # no 'squared' kwarg
    print(f"{name}  MAE: {mae:.2f}°F   RMSE: {rmse:.2f}°F")

pred_tmax = m_tmax.predict(Xva_tmax)
pred_tmin = m_tmin.predict(Xva_tmin)

print("\nOut-of-sample (last 20%)")
eval_model(yva_tmax, pred_tmax, "Next-day Tmax")
eval_model(yva_tmin, pred_tmin, "Next-day Tmin")

# Persistence baselines
persistence_tmax = daily["tmax_lag1"].values[split_idx:]
persistence_tmin = daily["tmin_lag1"].values[split_idx:]

print("\nPersistence baselines")
eval_model(yva_tmax, persistence_tmax, "Tmax persistence")
eval_model(yva_tmin, persistence_tmin, "Tmin persistence")

# ---------- Save models + feature list ----------
os.makedirs("models", exist_ok=True)
joblib.dump(m_tmax, "models/xgb_tmax_D1_austin.pkl")
joblib.dump(m_tmin, "models/xgb_tmin_D1_austin.pkl")
pd.Series(feature_cols).to_csv("models/xgb_temp_features.txt", index=False)
print("\nSaved models to models/xgb_tmax_D1_austin.pkl and models/xgb_tmin_D1_austin.pkl")

# ---------- Quick preview ----------
preview = pd.DataFrame({
    "date": daily.loc[split_idx:, "date"].values,
    "tmax_true": yva_tmax, "tmax_pred": pred_tmax,
    "tmin_true": yva_tmin, "tmin_pred": pred_tmin
}).head(10)
print("\nPreview:\n", preview)
