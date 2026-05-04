from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import sys

import pandas as pd


# Project Import Setup
# This supports running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import (
    CLEANED_WEATHER_DIR,
    LIVE_FEATURE_ROWS_DIR,
    OPEN_METEO_FORECAST_DIR,
    ensure_project_dirs,
)


def load_latest_open_meteo_forecast(region_key: str) -> Dict[str, Any]:
    """
    Load the latest raw Open-Meteo forecast JSON for the selected region.
    """
    input_path = OPEN_METEO_FORECAST_DIR / f"open_meteo_forecast_{region_key}_latest.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Latest Open-Meteo forecast file not found: {input_path}. "
            "Run src/ingestion/pull_open_meteo.py first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def flatten_hourly_forecast(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert Open-Meteo hourly forecast arrays into a clean row-based DataFrame.
    """
    hourly = data.get("hourly", {})

    if not hourly:
        raise ValueError("Missing 'hourly' block in Open-Meteo response.")

    if "time" not in hourly:
        raise KeyError("Missing hourly time values in Open-Meteo response.")

    row_count = len(hourly["time"])
    rows = []

    for index in range(row_count):
        row = {}

        for column_name, values in hourly.items():
            if not isinstance(values, list):
                continue

            if len(values) != row_count:
                raise ValueError(
                    f"Column '{column_name}' has {len(values)} values, "
                    f"but expected {row_count}."
                )

            row[column_name] = values[index]

        rows.append(row)

    forecast_df = pd.DataFrame(rows)

    forecast_df["timestamp"] = pd.to_datetime(forecast_df["time"])
    forecast_df = forecast_df.drop(columns=["time"])
    forecast_df = forecast_df.sort_values("timestamp").reset_index(drop=True)

    return forecast_df


def add_weather_features(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather features that are useful for load forecasting and drift detection.
    """
    df = forecast_df.copy()

    if "temperature_2m" in df.columns:
        df["heating_degree_hours"] = (65 - df["temperature_2m"]).clip(lower=0)
        df["cooling_degree_hours"] = (df["temperature_2m"] - 65).clip(lower=0)
        df["temperature_delta_1h"] = df["temperature_2m"].diff()

    if "relative_humidity_2m" in df.columns:
        df["humidity_delta_1h"] = df["relative_humidity_2m"].diff()

    if "precipitation_probability" in df.columns:
        df["is_precipitation_expected"] = (
            df["precipitation_probability"].fillna(0) >= 50
        ).astype(int)

    if "precipitation" in df.columns:
        df["has_measurable_precipitation"] = (
            df["precipitation"].fillna(0) > 0
        ).astype(int)

    if "wind_speed_10m" in df.columns:
        df["is_high_wind"] = (df["wind_speed_10m"].fillna(0) >= 25).astype(int)

    if "cloud_cover" in df.columns:
        df["is_cloudy"] = (df["cloud_cover"].fillna(0) >= 70).astype(int)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def select_next_hour_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the next available future hour from the forecast table.
    """
    now = datetime.now()
    future_rows = feature_df[feature_df["timestamp"] > pd.Timestamp(now)]

    if future_rows.empty:
        raise ValueError("No future hourly rows found in the forecast table.")

    next_hour_row = future_rows.head(1).copy()

    return next_hour_row


def save_processed_forecast(
    feature_df: pd.DataFrame,
    next_hour_df: pd.DataFrame,
    region_key: str,
) -> Dict[str, Path]:
    """
    Save the cleaned forecast table and next-hour feature row.
    """
    CLEANED_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_FEATURE_ROWS_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_output_path = (
        CLEANED_WEATHER_DIR / f"open_meteo_forecast_{region_key}_latest.csv"
    )

    next_hour_output_path = (
        LIVE_FEATURE_ROWS_DIR / f"open_meteo_next_hour_features_{region_key}.csv"
    )

    feature_df.to_csv(cleaned_output_path, index=False)
    next_hour_df.to_csv(next_hour_output_path, index=False)

    return {
        "cleaned_forecast": cleaned_output_path,
        "next_hour_features": next_hour_output_path,
    }


def print_processing_summary(
    feature_df: pd.DataFrame,
    next_hour_df: pd.DataFrame,
    output_paths: Dict[str, Path],
) -> None:
    """
    Print a short summary of the processed forecast data.
    """
    next_hour_record = next_hour_df.iloc[0].to_dict()

    print("Open-Meteo Forecast Processing Complete")
    print("Processed Rows:", len(feature_df))
    print("Processed Columns:", len(feature_df.columns))
    print("First Timestamp:", feature_df["timestamp"].min())
    print("Last Timestamp:", feature_df["timestamp"].max())
    print("Next-Hour Timestamp:", next_hour_record.get("timestamp"))
    print("Next-Hour Temperature:", next_hour_record.get("temperature_2m"))
    print("Next-Hour Humidity:", next_hour_record.get("relative_humidity_2m"))
    print("Cleaned Forecast File:", output_paths["cleaned_forecast"])
    print("Next-Hour Feature File:", output_paths["next_hour_features"])


def main() -> None:
    """
    Process the latest Open-Meteo forecast into clean weather feature files.
    """
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    raw_forecast_data = load_latest_open_meteo_forecast(region_key)
    hourly_forecast_df = flatten_hourly_forecast(raw_forecast_data)
    feature_df = add_weather_features(hourly_forecast_df)
    next_hour_df = select_next_hour_features(feature_df)

    output_paths = save_processed_forecast(
        feature_df=feature_df,
        next_hour_df=next_hour_df,
        region_key=region_key,
    )

    print_processing_summary(
        feature_df=feature_df,
        next_hour_df=next_hour_df,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()