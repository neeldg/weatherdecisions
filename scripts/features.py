import pandas as pd
import numpy as np

NUM_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "wind_speed_10m",
    "apparent_temperature",
]
CAT_COLS = ["weather_code"]  # WMO codes

def _ensure_numeric(df):
    df = df.copy()
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[date_col])
    out = df.copy()
    out["doy"] = ts.dt.dayofyear
    out["sin_doy"] = np.sin(2*np.pi*out["doy"]/365.25)
    out["cos_doy"] = np.cos(2*np.pi*out["doy"]/365.25)
    return out

def add_lags_rolls(df: pd.DataFrame, cols, lags=(1,2,3,7), rolls=(7,14,30)) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns: 
            continue
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
        for W in rolls:
            out[f"{c}_roll{W}"] = out[c].rolling(W, min_periods=W).mean()
            out[f"{c}_std{W}"]  = out[c].rolling(W, min_periods=W).std()
    return out

def one_hot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vals = df[col].astype("Int64").astype("string").fillna("nan")
    oh = pd.get_dummies(vals, prefix=col, dtype=int)
    return pd.concat([df.drop(columns=[col]), oh], axis=1)

def build_features(raw: pd.DataFrame, date_col: str, target_col: str):
    """
    Returns:
      df: feature-engineered frame with target shifted to D+1
      feature_cols: list of columns to feed the model
    """
    df = raw.sort_values(date_col).reset_index(drop=True)
    df = _ensure_numeric(df)
    df = add_time_features(df, date_col)

    df = add_lags_rolls(df, [c for c in NUM_COLS if c in df.columns], lags=(1,2,3,7), rolls=(7,14,30))

    # Useful lag for categorical weather pattern
    if "weather_code" in df.columns:
        df["weather_code_lag1"] = df["weather_code"].shift(1)

    # Shift target to "tomorrow"
    df[target_col] = df[target_col].shift(-1)

    # One-hot encode categorical columns present
    for col in ["weather_code", "weather_code_lag1"]:
        if col in df.columns:
            df = one_hot(df, col)

    # Drop rows with NA introduced by lags/rolls/shift
    df = df.dropna().reset_index(drop=True)

    # Feature list: numeric columns except date & target
    feature_cols = [c for c in df.columns
                    if c not in [date_col, target_col] and pd.api.types.is_numeric_dtype(df[c])]

    return df, feature_cols

def build_latest_features(history_df: pd.DataFrame, date_col: str, feature_cols: list, target_col: str):
    """
    For live prediction: recompute the SAME features on full history and
    return the last row aligned to feature_cols.
    """
    df_feat, feats = build_features(history_df, date_col, target_col)
    # We built target as shifted(-1); the last engineered row corresponds to predicting next day from "today-1".
    X_last = df_feat[feats].iloc[[-1]]
    # Align to saved training features
    X_last = X_last.reindex(columns=feature_cols, fill_value=0)
    return X_last
