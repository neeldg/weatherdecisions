import pandas as pd

def persistence_on_val(df_val: pd.DataFrame, target_col: str):
    """
    After feature build, target_col is temp at t+1; persistence predicts y_hat = temp at t.
    The original (unshifted) temp is not present here, but we can use the engineered
    feature 'temperature_2m_lag1' as a strong proxy; alternatively, ensure the raw col
    existed before shift (handled by pipeline).
    """
    # If the pipeline kept original 'temperature_2m' (it did, but shifted target replaced it),
    # the best "tomorrow = today" proxy is the 1-day lag of tomorrow, i.e., today's temp:
    if "temperature_2m" in df_val.columns:
        # Note: here df_val[target] is tomorrow's temp. A strict persistence would require access
        # to today's temp column; given we retained it before shift, this works.
        return df_val["temperature_2m"]
    elif "temperature_2m_lag1" in df_val.columns:
        return df_val["temperature_2m_lag1"]
    else:
        raise ValueError("No suitable column for persistence baseline found.")
