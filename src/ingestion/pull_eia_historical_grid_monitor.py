from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config, get_source_config, load_data_sources_config
from src.utils.paths import EIA_DATA_DIR, CLEANED_LOAD_DIR, ensure_project_dirs


def get_eia_api_key(eia_config: Dict[str, Any]) -> str:
    api_key_env_var = eia_config.get("api_key_env_var", "EIA_API_KEY")
    api_key = os.getenv(api_key_env_var)

    if not api_key:
        raise ValueError(
            f"Missing EIA API key. Add {api_key_env_var}=your_key_here to your .env file."
        )

    return api_key


def get_historical_backfill_dates() -> tuple[str, str]:
    data_sources_config = load_data_sources_config()
    historical_backfill = data_sources_config.get("ingestion", {}).get("historical_backfill", {})

    start_date = historical_backfill.get("default_start_date", "2023-01-01")
    end_date = historical_backfill.get("default_end_date", "2025-12-31")

    return start_date, end_date


def build_eia_grid_monitor_url(eia_config: Dict[str, Any]) -> str:
    base_url = eia_config.get("base_url")
    route = eia_config.get("electricity_rto", {}).get("route")

    if not base_url:
        raise KeyError("Missing EIA base_url in config/data_sources.yaml.")

    if not route:
        raise KeyError("Missing EIA electricity_rto route in config/data_sources.yaml.")

    return f"{base_url}/{route}"


def build_eia_historical_params(
    eia_config: Dict[str, Any],
    region_config: Dict[str, Any],
    api_key: str,
    start_date: str,
    end_date: str,
    offset: int,
    length: int,
) -> Dict[str, Any]:
    query_defaults = eia_config.get("query_defaults", {})
    balancing_authority = region_config.get(
        "balancing_authority_code",
        query_defaults.get("balancing_authority", "PJM"),
    )

    params = {
        "api_key": api_key,
        "frequency": query_defaults.get("frequency", "hourly"),
        "data[0]": "value",
        "facets[respondent][]": balancing_authority,
        "start": f"{start_date}T00",
        "end": f"{end_date}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": length,
    }

    return params


def fetch_eia_page(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()

    return response.json()


def get_eia_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    response_block = data.get("response", {})
    records = response_block.get("data", [])

    if not isinstance(records, list):
        raise ValueError("Unexpected EIA response format. Expected response.data to be a list.")

    return records


def get_total_record_count(data: Dict[str, Any]) -> int | None:
    response_block = data.get("response", {})
    total = response_block.get("total")

    if total is None:
        return None

    return int(total)


def save_raw_eia_page(
    data: Dict[str, Any],
    region_key: str,
    start_date: str,
    end_date: str,
    page_number: int,
) -> Path:
    EIA_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        EIA_DATA_DIR
        / f"eia_historical_grid_monitor_{region_key}_{start_date}_to_{end_date}_page_{page_number:03d}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return output_path


def clean_eia_records(raw_df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df = raw_df.copy()

    required_columns = ["period", "type", "value"]

    for column in required_columns:
        if column not in df.columns:
            raise KeyError(f"Missing required EIA column: {column}")

    df["timestamp_utc"] = pd.to_datetime(df["period"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp_utc"].dt.tz_convert(timezone).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp", "value"])
    df = df.sort_values(["timestamp", "type"]).reset_index(drop=True)

    if "respondent" not in df.columns:
        df["respondent"] = None

    if "respondent-name" not in df.columns:
        df["respondent-name"] = None

    if "value-units" not in df.columns:
        df["value-units"] = None

    selected_columns = [
        "timestamp",
        "timestamp_utc",
        "period",
        "respondent",
        "respondent-name",
        "type",
        "value",
        "value-units",
    ]

    available_columns = [column for column in selected_columns if column in df.columns]

    return df[available_columns]


def save_combined_historical_eia(
    clean_df: pd.DataFrame,
    region_key: str,
    start_date: str,
    end_date: str,
) -> Path:
    CLEANED_LOAD_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        CLEANED_LOAD_DIR
        / f"eia_historical_grid_monitor_{region_key}_{start_date}_to_{end_date}.csv"
    )

    latest_path = CLEANED_LOAD_DIR / f"eia_historical_grid_monitor_{region_key}_latest.csv"

    clean_df.to_csv(output_path, index=False)
    clean_df.to_csv(latest_path, index=False)

    return output_path


def print_historical_eia_summary(
    clean_df: pd.DataFrame,
    raw_paths: List[Path],
    cleaned_output_path: Path,
    total_record_count: int | None,
) -> None:
    print("EIA Historical Grid Monitor Pull Complete")
    print("Raw Page Files:", len(raw_paths))
    print("Expected Total Records:", total_record_count)
    print("Processed Rows:", len(clean_df))
    print("Processed Columns:", len(clean_df.columns))
    print("First Timestamp:", clean_df["timestamp"].min())
    print("Last Timestamp:", clean_df["timestamp"].max())
    print("Types:", sorted(clean_df["type"].dropna().unique().tolist()))
    print("Cleaned Historical EIA File:", cleaned_output_path)


def main() -> None:
    load_dotenv()
    ensure_project_dirs()

    region_config = get_region_config()
    eia_config = get_source_config("eia")

    region_key = region_config["region_key"]
    timezone = region_config.get("timezone", "America/New_York")
    start_date, end_date = get_historical_backfill_dates()

    api_key = get_eia_api_key(eia_config)
    url = build_eia_grid_monitor_url(eia_config)

    length = eia_config.get("query_defaults", {}).get("length", 5000)
    offset = 0
    page_number = 1
    total_record_count = None

    raw_paths = []
    all_records = []

    print("Starting EIA historical grid monitor pull")
    print("Region:", region_key)
    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("Page Length:", length)

    while True:
        print(f"Pulling EIA page {page_number}, offset {offset}")

        params = build_eia_historical_params(
            eia_config=eia_config,
            region_config=region_config,
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
            offset=offset,
            length=length,
        )

        data = fetch_eia_page(url=url, params=params)

        if total_record_count is None:
            total_record_count = get_total_record_count(data)

        records = get_eia_records(data)

        if not records:
            break

        raw_path = save_raw_eia_page(
            data=data,
            region_key=region_key,
            start_date=start_date,
            end_date=end_date,
            page_number=page_number,
        )

        raw_paths.append(raw_path)
        all_records.extend(records)

        if len(records) < length:
            break

        offset += length
        page_number += 1

        if total_record_count is not None and offset >= total_record_count:
            break

        time.sleep(0.25)

    if not all_records:
        raise ValueError("No EIA historical records returned.")

    raw_df = pd.DataFrame(all_records)
    clean_df = clean_eia_records(raw_df, timezone)

    clean_df = clean_df.drop_duplicates(
        subset=["timestamp", "respondent", "type"],
        keep="last",
    )

    clean_df = clean_df.sort_values(["timestamp", "type"]).reset_index(drop=True)

    cleaned_output_path = save_combined_historical_eia(
        clean_df=clean_df,
        region_key=region_key,
        start_date=start_date,
        end_date=end_date,
    )

    print_historical_eia_summary(
        clean_df=clean_df,
        raw_paths=raw_paths,
        cleaned_output_path=cleaned_output_path,
        total_record_count=total_record_count,
    )


if __name__ == "__main__":
    main()