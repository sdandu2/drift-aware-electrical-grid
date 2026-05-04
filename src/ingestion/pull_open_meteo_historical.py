from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
import json
import sys
import time

import pandas as pd
import requests


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config, get_source_config, load_data_sources_config
from src.utils.paths import OPEN_METEO_HISTORICAL_DIR, CLEANED_WEATHER_DIR, ensure_project_dirs


def get_historical_backfill_dates() -> tuple[str, str, int]:
    data_sources_config = load_data_sources_config()
    historical_backfill = data_sources_config.get("ingestion", {}).get("historical_backfill", {})

    start_date = historical_backfill.get("default_start_date", "2023-01-01")
    end_date = historical_backfill.get("default_end_date", "2025-12-31")
    max_days_per_request = historical_backfill.get("max_days_per_request", 31)

    return start_date, end_date, max_days_per_request


def build_date_chunks(
    start_date: str,
    end_date: str,
    max_days_per_request: int,
) -> List[tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    if start > end:
        raise ValueError("Historical start date cannot be after end date.")

    chunks = []
    current_start = start

    while current_start <= end:
        current_end = min(current_start + timedelta(days=max_days_per_request - 1), end)
        chunks.append((current_start.isoformat(), current_end.isoformat()))
        current_start = current_end + timedelta(days=1)

    return chunks


def build_open_meteo_historical_params(
    region_config: Dict[str, Any],
    open_meteo_config: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    weather_points = region_config.get("weather_points", {})

    if "primary" not in weather_points:
        raise KeyError("Missing primary weather point in region configuration.")

    primary_point = weather_points["primary"]
    historical_config = open_meteo_config.get("historical", {})
    query_defaults = historical_config.get("query_defaults", {})
    hourly_variables = historical_config.get("hourly_variables", [])

    params = {
        "latitude": primary_point["latitude"],
        "longitude": primary_point["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": query_defaults.get("timezone", region_config.get("timezone")),
        "temperature_unit": query_defaults.get("temperature_unit", "fahrenheit"),
        "wind_speed_unit": query_defaults.get("wind_speed_unit", "mph"),
        "precipitation_unit": query_defaults.get("precipitation_unit", "inch"),
    }

    return params


def fetch_open_meteo_historical_chunk(
    open_meteo_config: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    historical_config = open_meteo_config.get("historical", {})
    base_url = historical_config.get("base_url")

    if not base_url:
        raise KeyError("Missing Open-Meteo historical base_url in data_sources.yaml.")

    response = requests.get(base_url, params=params, timeout=60)
    response.raise_for_status()

    return response.json()


def save_raw_historical_chunk(
    data: Dict[str, Any],
    region_key: str,
    start_date: str,
    end_date: str,
) -> Path:
    OPEN_METEO_HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OPEN_METEO_HISTORICAL_DIR
        / f"open_meteo_historical_{region_key}_{start_date}_to_{end_date}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return output_path


def flatten_hourly_historical_response(data: Dict[str, Any]) -> pd.DataFrame:
    hourly = data.get("hourly", {})

    if not hourly:
        raise ValueError("Missing hourly block in Open-Meteo historical response.")

    if "time" not in hourly:
        raise KeyError("Missing hourly time values in Open-Meteo historical response.")

    row_count = len(hourly["time"])
    rows = []

    for index in range(row_count):
        row = {}

        for column_name, values in hourly.items():
            if not isinstance(values, list):
                continue

            if len(values) != row_count:
                raise ValueError(
                    f"Column '{column_name}' has {len(values)} values, expected {row_count}."
                )

            row[column_name] = values[index]

        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.drop(columns=["time"])
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def add_historical_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    df = weather_df.copy()

    if "temperature_2m" in df.columns:
        df["heating_degree_hours"] = (65 - df["temperature_2m"]).clip(lower=0)
        df["cooling_degree_hours"] = (df["temperature_2m"] - 65).clip(lower=0)
        df["temperature_delta_1h"] = df["temperature_2m"].diff()
        df["temperature_lag_1h"] = df["temperature_2m"].shift(1)
        df["temperature_lag_24h"] = df["temperature_2m"].shift(24)

    if "relative_humidity_2m" in df.columns:
        df["humidity_delta_1h"] = df["relative_humidity_2m"].diff()
        df["humidity_lag_1h"] = df["relative_humidity_2m"].shift(1)
        df["humidity_lag_24h"] = df["relative_humidity_2m"].shift(24)

    if "precipitation" in df.columns:
        df["has_measurable_precipitation"] = (df["precipitation"].fillna(0) > 0).astype(int)

    if "wind_speed_10m" in df.columns:
        df["is_high_wind"] = (df["wind_speed_10m"].fillna(0) >= 25).astype(int)

    if "cloud_cover" in df.columns:
        df["is_cloudy"] = (df["cloud_cover"].fillna(0) >= 70).astype(int)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def save_combined_historical_weather(
    weather_df: pd.DataFrame,
    region_key: str,
    start_date: str,
    end_date: str,
) -> Path:
    CLEANED_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        CLEANED_WEATHER_DIR
        / f"open_meteo_historical_{region_key}_{start_date}_to_{end_date}.csv"
    )

    latest_path = CLEANED_WEATHER_DIR / f"open_meteo_historical_{region_key}_latest.csv"

    weather_df.to_csv(output_path, index=False)
    weather_df.to_csv(latest_path, index=False)

    return output_path


def print_historical_weather_summary(
    weather_df: pd.DataFrame,
    raw_paths: List[Path],
    cleaned_output_path: Path,
) -> None:
    print("Open-Meteo Historical Pull Complete")
    print("Raw Chunk Files:", len(raw_paths))
    print("Processed Rows:", len(weather_df))
    print("Processed Columns:", len(weather_df.columns))
    print("First Timestamp:", weather_df["timestamp"].min())
    print("Last Timestamp:", weather_df["timestamp"].max())
    print("Cleaned Historical Weather File:", cleaned_output_path)


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    open_meteo_config = get_source_config("open_meteo")

    region_key = region_config["region_key"]
    start_date, end_date, max_days_per_request = get_historical_backfill_dates()

    chunks = build_date_chunks(
        start_date=start_date,
        end_date=end_date,
        max_days_per_request=max_days_per_request,
    )

    raw_paths = []
    chunk_dataframes = []

    print("Starting Open-Meteo historical pull")
    print("Region:", region_key)
    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("Chunks:", len(chunks))

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"Pulling chunk {chunk_index}/{len(chunks)}: {chunk_start} to {chunk_end}")

        params = build_open_meteo_historical_params(
            region_config=region_config,
            open_meteo_config=open_meteo_config,
            start_date=chunk_start,
            end_date=chunk_end,
        )

        data = fetch_open_meteo_historical_chunk(
            open_meteo_config=open_meteo_config,
            params=params,
        )

        raw_path = save_raw_historical_chunk(
            data=data,
            region_key=region_key,
            start_date=chunk_start,
            end_date=chunk_end,
        )

        chunk_df = flatten_hourly_historical_response(data)

        raw_paths.append(raw_path)
        chunk_dataframes.append(chunk_df)

        time.sleep(0.25)

    combined_df = pd.concat(chunk_dataframes, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

    feature_df = add_historical_weather_features(combined_df)

    cleaned_output_path = save_combined_historical_weather(
        weather_df=feature_df,
        region_key=region_key,
        start_date=start_date,
        end_date=end_date,
    )

    print_historical_weather_summary(
        weather_df=feature_df,
        raw_paths=raw_paths,
        cleaned_output_path=cleaned_output_path,
    )


if __name__ == "__main__":
    main()