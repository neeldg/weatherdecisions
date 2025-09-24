import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# ----------------------
# Setup cached+retrying client
# ----------------------
cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ----------------------
# Request params
# ----------------------
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 30.2672,
    "longitude": -97.7431,
    "start_date": "2000-01-01",
    "end_date": "2025-09-23",
    # ORDER HERE MUST MATCH THE INDEXING BELOW
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "wind_speed_10m",
        "apparent_temperature",
        "weather_code",
    ],
    # Units + local timezone
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
    "timezone": "America/Chicago",
}

responses = openmeteo.weather_api(url, params=params)

# ----------------------
# Process first (and only) location
# ----------------------
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"UTC offset (s): {response.UtcOffsetSeconds()}")
print("Vars:", params["hourly"], "| TZ:", params["timezone"])

# ----------------------
# Hourly block
# ----------------------
hourly = response.Hourly()

# Build the hourly time index first
time_index = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left",
).tz_convert(params["timezone"])  # convert to local TZ

# Extract variables in the SAME ORDER as requested
var_names = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "wind_speed_10m",
    "apparent_temperature",
    "weather_code",
]
arrays = [hourly.Variables(i).ValuesAsNumpy() for i in range(len(var_names))]

# Quick length sanity checks
for name, arr in zip(var_names, arrays):
    assert len(arr) == len(time_index), f"Length mismatch for {name}: {len(arr)} vs {len(time_index)}"

# Assemble DataFrame
hourly_data = {"date": time_index}
for name, arr in zip(var_names, arrays):
    hourly_data[name] = arr

hourly_df = pd.DataFrame(hourly_data)

# Cleaner dtypes
if "weather_code" in hourly_df.columns:
    hourly_df["weather_code"] = hourly_df["weather_code"].astype("int16", errors="ignore")

# ----------------------
# NaN cleanup (inline)
# ----------------------
# Treat missing precip as 0; drop any remaining NAs (e.g., boundary/DST gaps)
if "precipitation" in hourly_df.columns:
    hourly_df["precipitation"] = hourly_df["precipitation"].fillna(0.0)
hourly_df = hourly_df.dropna().reset_index(drop=True)

print("NA counts after cleanup:\n", hourly_df.isna().sum())
print("\nPreview:\n", hourly_df.head())

# ----------------------
# Save outputs
# ----------------------
csv_path = "data/openmeteo_archive_austin_2000_2025.csv"
hourly_df.to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path}")

# Parquet (optional, faster to load). Requires 'pyarrow' or 'fastparquet'.
try:
    pq_path = "data/openmeteo_archive_austin_2000_2025.parquet"
    hourly_df.to_parquet(pq_path, index=False)
    print(f"Saved Parquet: {pq_path}")
except Exception as e:
    print("Parquet save skipped (install 'pyarrow' or 'fastparquet' to enable). Reason:", str(e))
